#!/usr/bin/env python3
"""
Infinite Cartridge System - Brain-Inspired Modular Neural Network
=================================================================
Millions of tiny specialized "cartridges" (~100KB) share frozen universal "stem" layers.
Learned router (prefrontal cortex) selects 2-4 experts per query.
Global workspace binds outputs. Hebbian learning strengthens successful pathways.

Optimized for Apple Silicon: cartridges fit in L2 cache for streaming inference.
"""

import os, uuid, json, time, queue, asyncio, logging, threading, traceback, random, shutil, re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
from contextlib import asynccontextmanager
import requests

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger(__name__)

# Hard requirement with clear error
try:
    import websockets
    import sse_starlette
except ImportError as e:
    log.critical("❌ MISSING REQUIRED PACKAGES")
    log.critical("   Install with: pip install websockets sse-starlette")
    log.critical(f"   Error: {e}")
    raise RuntimeError("Missing critical dependencies")

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass 
class Config:
    dim: int = 256
    stem_depth: int = 4
    heads: int = 8
    adapter_rank: int = 16
    
    cart_vocab: int = 256           # Reduced from 1024 - more cartridges!
    max_carts: int = 10000
    active_k: int = 4
    workspace_slots: int = 4
    
    batch: int = 32
    seq_len: int = 64
    lr: float = 3e-4
    
    library: str = './cartridge_library'
    state_file: str = './cartridge_library/state.json'
    spawn_interval: int = 50        # More frequent spawn checks
    spawn_min_tokens: int = 30      # Lower threshold for spawning
    save_interval: int = 2000
    
    # Round-robin training to ensure all cartridges get trained
    round_robin_interval: int = 20  # Switch cartridge every N steps
    
    # Gutenberg downloader settings
    gutenberg_enabled: bool = True
    gutenberg_download_delay: int = 60  # seconds between downloads
    gutenberg_max_books: int = 1000
    gutenberg_min_length: int = 10000  # minimum characters to keep

CFG = Config()
os.makedirs(CFG.library, exist_ok=True)


class BatchTuner:
    """
    Intelligent batch optimizer with 30-second moving average
    and histogram tracking for stable, data-driven decisions.
    """
    def __init__(self, initial: int = 8, min_b: int = 8, max_b: int = 256):
        self.batch = initial
        self.min_b = min_b
        self.max_b = max_b
        self.best_tps = 0.0
        self.best_batch = initial
        self.error_count = 0
        self.last_error_batch = 999999
        
        # Time-based tracking (30-second window)
        self.tps_window = []  # List of (timestamp, tps) tuples
        self.window_seconds = 30
        self.min_samples = 5  # Minimum measurements needed
        
        # Histogram/stats tracking
        self.tps_history = []  # Last 100 raw TPS values
        self.max_history = 100
        self.volatility_threshold = 0.15  # 15% volatility limit
        
        # Confirmation system
        self.pending_change = None  # (batch_size, reason)
        self.confirmation_needed = 3  # Need 3 consistent readings
        self.confirmation_count = 0
        
        # Growth parameters (moderate aggression)
        self.stability_threshold = 10  # ~10 steps of stability
        self.growth_factor = 1.15
        self.improvement_threshold = 1.02  # 2% improvement
        self.stability_count = 0  # Track stability
        
    def _clean_old_samples(self):
        """Remove samples older than window_seconds"""
        now = time.time()
        cutoff = now - self.window_seconds
        self.tps_window = [(ts, val) for ts, val in self.tps_window if ts > cutoff]
    
    def get_moving_average_tps(self) -> float:
        """Calculate TPS average over last 30 seconds"""
        self._clean_old_samples()
        
        if len(self.tps_window) < self.min_samples:
            return 0.0
        
        recent_tps = [val for _, val in self.tps_window]
        return sum(recent_tps) / len(recent_tps)
    
    def get_volatility(self) -> float:
        """Calculate coefficient of variation (volatility %)"""
        self._clean_old_samples()
        
        if len(self.tps_window) < 2:
            return 0.0
        
        recent_tps = [val for _, val in self.tps_window]
        mean = sum(recent_tps) / len(recent_tps)
        
        if mean == 0:
            return 0.0
        
        variance = sum((x - mean) ** 2 for x in recent_tps) / len(recent_tps)
        std_dev = variance ** 0.5
        return std_dev / mean  # Coefficient of variation
    
    def update(self, step_time: float) -> int:
        """Update with time-based averaging and confirmation"""
        now = time.time()
        tps = (self.batch * CFG.seq_len) / max(step_time, 0.001)
        
        # Store sample
        self.tps_window.append((now, tps))
        self.tps_history.append(tps)
        self.tps_history = self.tps_history[-self.max_history:]
        
        # Get averaged metrics
        avg_tps = self.get_moving_average_tps()
        volatility = self.get_volatility()
        
        # Skip if insufficient data
        if avg_tps == 0:
            return self.batch
        
        # Track best performance (using moving average)
        if avg_tps > self.best_tps * self.improvement_threshold:
            self.best_tps = avg_tps
            self.best_batch = self.batch
            self.stability_count = 0
            log.debug(f"📈 New best avg TPS: {avg_tps:.1f} (batch {self.batch})")
        else:
            self.stability_count += 1
        
        # Check if system is stable enough for a change
        is_stable = volatility < self.volatility_threshold
        is_stable_long = self.stability_count > self.stability_threshold
        
        # If stable, check for growth opportunity
        if is_stable and is_stable_long and self.batch < self.max_b:
            new_batch = min(self.max_b, int(self.batch * self.growth_factor))
            
            # Check confirmation
            if self.pending_change == new_batch:
                self.confirmation_count += 1
                if self.confirmation_count >= self.confirmation_needed:
                    if new_batch < self.last_error_batch:
                        old_batch = self.batch
                        self.batch = new_batch
                        self.stability_count = 0
                        self.pending_change = None
                        self.confirmation_count = 0
                        log.info(f"📊 BatchTuner: {old_batch} → {new_batch} "
                               f"| Avg TPS: {avg_tps:.1f} | Vol: {volatility:.2%}")
                        return self.batch
            else:
                # Start new confirmation sequence
                self.pending_change = new_batch
                self.confirmation_count = 1
                log.debug(f"📊 Considering batch {new_batch} (need {self.confirmation_needed} confirmations)")
        else:
            # Reset if conditions aren't met
            self.pending_change = None
            self.confirmation_count = 0
        
        # Periodically log detailed statistics
        if len(self.tps_window) % 50 == 0 and len(self.tps_window) >= self.min_samples:
            log.info(f"📊 BatchTuner: window={len(self.tps_window)} samples, "
                     f"avg_tps={avg_tps:.1f}, vol={volatility:.1%}, "
                     f"stable={self.stability_count}, pending={self.pending_change}, "
                     f"confirmations={self.confirmation_count}")
        
        return self.batch
    
    def on_error(self) -> int:
        """Handle OOM or compute errors"""
        old_batch = self.batch
        self.batch = max(self.min_b, self.batch // 2)
        self.error_count += 1
        self.last_error_batch = old_batch
        
        # Reset confirmations
        self.pending_change = None
        self.confirmation_count = 0
        
        log.warning(f"⚠️  BatchTuner: {old_batch} → {self.batch} (error) | Max: {self.max_b}")
        return self.batch
    
    def get_histogram(self, bins: int = 10) -> dict:
        """Return TPS histogram data for visualization"""
        if not self.tps_history:
            return {"bins": [], "counts": [], "current": 0, "best": 0, "mean": 0}
        
        # Calculate bins
        min_tps = min(self.tps_history)
        max_tps = max(self.tps_history)
        
        if max_tps == min_tps:
            return {
                "bins": [min_tps],
                "counts": [len(self.tps_history)],
                "current": self.batch,
                "best": self.best_batch,
                "mean": min_tps
            }
        
        bin_width = (max_tps - min_tps) / bins
        hist_counts = [0] * bins
        
        for tps in self.tps_history:
            bin_idx = min(int((tps - min_tps) / bin_width), bins - 1)
            hist_counts[bin_idx] += 1
        
        return {
            "bins": [min_tps + (i + 0.5) * bin_width for i in range(bins)],
            "counts": hist_counts,
            "current": self.batch,
            "best": self.best_batch,
            "mean": sum(self.tps_history) / len(self.tps_history),
            "volatility": self.get_volatility()
        }
    
    def status(self) -> dict:
        """Get detailed status for monitoring"""
        self._clean_old_samples()
        return {
            "current": self.batch, 
            "best": self.best_batch, 
            "best_tps": round(self.best_tps, 1),
            "max": self.max_b,
            "errors": self.error_count,
            "window_samples": len(self.tps_window),
            "pending_change": self.pending_change,
            "confirmations": self.confirmation_count,
            "volatility": round(self.get_volatility() * 100, 1),
            "volatility_threshold": round(self.volatility_threshold * 100, 1)
        }
    
    def save(self, path: str = None):
        """Persist tuner state to disk"""
        if path is None:
            path = os.path.join(CFG.library, 'tuner_state.json')
        
        data = {
            'batch': self.batch,
            'best_batch': self.best_batch,
            'best_tps': self.best_tps,
            'max_b': self.max_b,
            'last_error_batch': self.last_error_batch,
            'error_count': self.error_count,
            'last_save': time.time()
        }
        
        try:
            tmp_path = path + '.tmp'
            with open(tmp_path, 'w') as f:
                json.dump(data, f)
            Path(tmp_path).replace(path)  # Atomic write
            log.debug(f"💾 BatchTuner state saved: batch={self.batch}, best={self.best_batch}")
        except Exception as e:
            log.error(f"BatchTuner save failed: {e}")
    
    def load(self, path: str = None):
        """Restore tuner state from disk"""
        if path is None:
            path = os.path.join(CFG.library, 'tuner_state.json')
        
        if not os.path.exists(path):
            log.debug("ℹ️  No saved BatchTuner state found, starting fresh")
            return
        
        try:
            with open(path) as f:
                data = json.load(f)
            
            # Restore critical state
            self.batch = data.get('batch', self.batch)
            self.best_batch = data.get('best_batch', self.best_batch)
            self.best_tps = data.get('best_tps', 0.0)
            self.max_b = data.get('max_b', self.max_b)
            self.last_error_batch = data.get('last_error_batch', 999999)
            self.error_count = data.get('error_count', 0)
            
            log.info(f"📂 BatchTuner loaded: batch={self.batch}, best={self.best_batch}, "
                    f"best_tps={self.best_tps:.1f}, max={self.max_b}")
        except Exception as e:
            log.warning(f"BatchTuner load failed: {e}, using defaults")


TUNER = BatchTuner(initial=8, min_b=8, max_b=256)
GPU_LOCK = threading.Lock()

# =============================================================================
# NEURAL PRIMITIVES
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.w = mx.ones((d,))
    
    def __call__(self, x):
        return x * mx.rsqrt(mx.mean(x*x, axis=-1, keepdims=True) + self.eps) * self.w


class SwiGLU(nn.Module):
    def __init__(self, d: int, mult: int = 4):
        super().__init__()
        h = d * mult
        self.w1 = nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(h, d, bias=False)  
        self.w3 = nn.Linear(d, h, bias=False)
    
    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, d: int, h: int):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = nn.MultiHeadAttention(d, h)
        self.norm2 = RMSNorm(d)
        self.mlp = SwiGLU(d)
    
    def __call__(self, x, mask=None):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, mask=mask)
        x = x + self.mlp(self.norm2(x))
        return x


# =============================================================================
# STEM (Frozen Universal Backbone)
# =============================================================================

class Stem(nn.Module):
    def __init__(self, vocab: int = 32000, d: int = 256, depth: int = 4, heads: int = 8):
        super().__init__()
        self.d = d
        self.embed = nn.Embedding(vocab, d)
        self.blocks = [Block(d, heads) for _ in range(depth)]
        self.norm = RMSNorm(d)
        self._mask_cache = {}
    
    def _get_mask(self, L: int) -> mx.array:
        if L not in self._mask_cache:
            self._mask_cache[L] = nn.MultiHeadAttention.create_additive_causal_mask(L).astype(mx.float16)
        return self._mask_cache[L]
    
    def __call__(self, x: mx.array, signal: mx.array = None) -> mx.array:
        h = self.embed(x)
        if signal is not None:
            h = h + signal.reshape(1, 1, -1)
        
        mask = self._get_mask(x.shape[1])
        for block in self.blocks:
            h = block(h, mask)
        return self.norm(h)


# =============================================================================
# CARTRIDGE (Specialized Expert ~100KB)
# =============================================================================

class LoRA(nn.Module):
    def __init__(self, d: int, r: int = 16):
        super().__init__()
        self.A = nn.Linear(d, r, bias=False)
        self.B = nn.Linear(r, d, bias=False)
        self.B.weight = mx.zeros_like(self.B.weight)
    
    def __call__(self, x):
        return x + self.B(self.A(x)) * 0.1


class Cartridge:
    __slots__ = ('id', 'stem', 'tokens', 'vocab_size', 'local_to_global', 'global_to_local',
                 '_lookup', 'lora', 'signal', 'head', 'created', 'last_used', 'steps', 
                 'strength', 'total_loss', 'loss_count', '_lock')
    
    def __init__(self, stem: Stem, tokens: set = None, uid: str = None):
        self.id = uid or uuid.uuid4().hex[:8]
        self.stem = stem
        self.tokens = set(tokens) if tokens else set()
        self.created = time.time()
        self.last_used = time.time()
        self.steps = 0
        self.strength = 1.0
        self.total_loss = 0.0
        self.loss_count = 0
        self._lock = threading.Lock()
        
        # Build vocab mapping
        self.local_to_global = [0, 1] + sorted(self.tokens)
        self.global_to_local = {g: i for i, g in enumerate(self.local_to_global)}
        self.vocab_size = len(self.local_to_global)
        
        # Vectorized lookup
        maxg = max(self.local_to_global) if self.local_to_global else 1
        self._lookup = np.ones(maxg + 1, dtype=np.int32)
        for g, l in self.global_to_local.items():
            self._lookup[g] = l
        
        # FIX: Initialize signal with RANDOM values, not zeros
        # This ensures router can differentiate cartridges
        self.signal = mx.random.normal((CFG.dim,), dtype=mx.float16) * 0.1
        self.lora = LoRA(CFG.dim, CFG.adapter_rank)
        self.head = nn.Linear(CFG.dim, self.vocab_size, bias=False)
        
        self.lora.set_dtype(mx.float16)
        self.head.set_dtype(mx.float16)
        mx.eval(self.lora.parameters(), self.head.parameters(), self.signal)
    
    @property
    def avg_loss(self) -> float:
        return self.total_loss / max(1, self.loss_count)
    
    def add_tokens(self, new_tokens: set):
        """Thread-safe token addition with atomic weight matrix expansion"""
        if not new_tokens:
            return
        
        with self._lock:
            added = new_tokens - self.tokens
            if not added:
                return
            
            self.tokens.update(added)
            
            # CRITICAL: Capture current weight matrix ATOMICALLY
            old_weight = self.head.weight
            actual_old_size = old_weight.shape[0]  # Use REAL shape, not cached value
            
            # Rebuild mappings
            self.local_to_global = [0, 1] + sorted(self.tokens)
            self.global_to_local = {g: i for i, g in enumerate(self.local_to_global)}
            self.vocab_size = len(self.local_to_global)
            
            # Extend lookup table
            maxg = max(self.local_to_global)
            if maxg >= len(self._lookup):
                new_lookup = np.ones(maxg + 1000, dtype=np.int32)
                new_lookup[:len(self._lookup)] = self._lookup
                self._lookup = new_lookup
            for g, l in self.global_to_local.items():
                self._lookup[g] = l
            
            # Expand head if needed
            if self.vocab_size > actual_old_size:
                # Use actual_old_size from the captured weight matrix
                new_weight = mx.zeros((self.vocab_size, CFG.dim), dtype=mx.float16)
                new_weight[:actual_old_size, :] = old_weight  # Now guaranteed to match
                
                # Initialize new weights
                new_weight[actual_old_size:, :] = mx.random.normal(
                    (self.vocab_size - actual_old_size, CFG.dim), dtype=mx.float16
                ) * 0.02
                
                self.head.weight = new_weight
                mx.eval(self.head.weight)
    
    def encode(self, global_ids) -> mx.array:
        arr = np.asarray(global_ids, dtype=np.int32)
        mask = arr >= len(self._lookup)
        result = np.where(mask, 1, self._lookup[np.minimum(arr, len(self._lookup)-1)])
        return mx.array(result)
    
    def forward(self, x: mx.array, workspace: mx.array = None) -> Tuple[mx.array, mx.array]:
        self.last_used = time.time()
        h = self.stem(x, self.signal)
        h = self.lora(h)
        if workspace is not None:
            h = h + workspace.reshape(1, 1, -1) * 0.1
        return self.head(h), h
    
    def train_step(self, x_global: np.ndarray, y_global: np.ndarray) -> float:
        # Extend lookup if needed
        maxv = max(x_global.max(), y_global.max())
        if maxv >= len(self._lookup):
            new_lookup = np.ones(maxv + 1000, dtype=np.int32)
            new_lookup[:len(self._lookup)] = self._lookup
            self._lookup = new_lookup
        
        x = mx.array(self._lookup[x_global])
        y = mx.array(self._lookup[y_global])
        
        def loss_fn(lora, head, sig):
            h = self.stem(x, sig)
            h = lora(h)
            logits = head(h)
            return nn.losses.cross_entropy(logits, y).mean()
        
        # Acquire lock during gradient computation
        with self._lock:
            loss, grads = mx.value_and_grad(loss_fn, argnums=(0, 1, 2))(self.lora, self.head, self.signal)
            
            lr = CFG.lr
            for (_, p), (_, g) in zip(tree_flatten(self.lora.parameters()), tree_flatten(grads[0])):
                p -= lr * g
            for (_, p), (_, g) in zip(tree_flatten(self.head.parameters()), tree_flatten(grads[1])):
                p -= lr * g
            self.signal -= lr * grads[2]
            
            mx.eval(self.lora.parameters(), self.head.parameters(), self.signal)
            self.steps += 1
        
        loss_val = float(loss)
        self.total_loss += loss_val
        self.loss_count += 1
        
        return loss_val
    
    def generate(self, tokens: List[int], max_new: int = 20, temp: float = 0.8) -> List[int]:
        curr = list(tokens)
        out = []
        
        for _ in range(max_new):
            ctx = curr[-CFG.seq_len:]
            x = self.encode(ctx).reshape(1, -1)
            logits, _ = self.forward(x)
            mx.eval(logits)
            
            probs = mx.softmax(logits[0, -1] / temp)
            idx = int(mx.random.categorical(mx.log(probs + 1e-10)))
            
            if idx <= 1:
                if len(out) >= 3:
                    break
                continue
            
            gid = self.local_to_global[idx] if idx < len(self.local_to_global) else 1
            curr.append(gid)
            out.append(gid)
        
        return out
    
    def save(self, path: str):
        folder = Path(path) / self.id
        folder.mkdir(parents=True, exist_ok=True)
        
        # Save weights
        params = {'signal': self.signal}
        for k, v in tree_flatten(self.lora.parameters()):
            params[f'lora.{k}'] = v
        for k, v in tree_flatten(self.head.parameters()):
            params[f'head.{k}'] = v
        mx.save_safetensors(str(folder / 'weights.safetensors'), params)
        
        # Atomic JSON write - write to temp then rename
        # FIX: Convert tokens to Python int (numpy int32 is not JSON serializable)
        # Convert all tokens to native Python int using recommended approach
        token_list = []
        for t in self.tokens:
            # Convert numpy integers to Python int (recommended approach)
            if isinstance(t, np.integer):
                token_list.append(int(t))
            else:
                token_list.append(int(t))
        
        meta = {
            'id': self.id, 
            'tokens': token_list, 
            'created': self.created,
            'steps': self.steps, 
            'strength': self.strength,
            'total_loss': self.total_loss,
            'loss_count': self.loss_count
        }
        tmp_path = folder / 'meta.json.tmp'
        final_path = folder / 'meta.json'
        with open(tmp_path, 'w') as f:
            json.dump(meta, f)
        tmp_path.replace(final_path)  # Atomic on POSIX
    
    @classmethod
    def load(cls, uid: str, stem: Stem, path: str) -> Optional['Cartridge']:
        folder = Path(path) / uid
        if not folder.exists():
            return None
        
        meta_path = folder / 'meta.json'
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            log.warning(f"Corrupted cartridge {uid}, removing: {e}")
            try:
                shutil.rmtree(folder)
            except Exception as rm_err:
                log.error(f"Failed to remove {uid}: {rm_err}")
            return None
        
        try:
            cart = cls(stem, set(meta.get('tokens', [])), uid)
            cart.created = meta.get('created', time.time())
            cart.steps = meta.get('steps', 0)
            cart.strength = meta.get('strength', 1.0)
            cart.total_loss = meta.get('total_loss', 0.0)
            cart.loss_count = meta.get('loss_count', 0)
            
            wpath = folder / 'weights.safetensors'
            if wpath.exists():
                W = mx.load(str(wpath))
                if 'signal' in W:
                    cart.signal = W['signal']
                if 'head.weight' in W:
                    cart.head.weight = W['head.weight']
                mx.eval(cart.signal, cart.head.weight)
            return cart
        except Exception as e:
            log.error(f"Load {uid} failed: {e}")
            return None


# =============================================================================
# ROUTER (Prefrontal Connector Hub)
# =============================================================================

class Router:
    def __init__(self, d: int):
        self.d = d
        self.signatures: Dict[str, mx.array] = {}
        self.strengths: Dict[str, float] = {}
        self.encoder = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.encoder.set_dtype(mx.float16)
        mx.eval(self.encoder.parameters())
    
    def register(self, cart: Cartridge):
        sig = cart.signal
        norm = mx.linalg.norm(sig)
        # FIX: Handle zero/near-zero signals
        if float(norm) < 1e-6:
            sig = mx.random.normal((self.d,), dtype=mx.float16) * 0.1
            cart.signal = sig
            mx.eval(cart.signal)
            norm = mx.linalg.norm(sig)
        self.signatures[cart.id] = sig / (norm + 1e-8)
        self.strengths[cart.id] = cart.strength
    
    def route(self, ctx: mx.array, k: int = 4) -> List[Tuple[str, float]]:
        if not self.signatures:
            return []
        
        q = self.encoder(ctx)
        q_norm = mx.linalg.norm(q)
        if float(q_norm) < 1e-6:
            # Return random ordering if query is degenerate
            items = list(self.signatures.keys())
            random.shuffle(items)
            return [(cid, 1.0) for cid in items[:k]]
        
        q = q / (q_norm + 1e-8)
        
        scores = {}
        for cid, sig in self.signatures.items():
            sim = float(mx.sum(q * sig))
            scores[cid] = sim * self.strengths.get(cid, 1.0)
        
        return sorted(scores.items(), key=lambda x: -x[1])[:k]
    
    def hebbian(self, cid: str, success: bool, amt: float = 0.01):
        if cid in self.strengths:
            if success:
                self.strengths[cid] = min(2.0, self.strengths[cid] + amt)
            else:
                self.strengths[cid] = max(0.1, self.strengths[cid] - amt)
    
    def update_sig(self, cart: Cartridge, ctx: mx.array, loss: float):
        if cart.id not in self.signatures:
            self.register(cart)
            return
        
        old = self.signatures[cart.id]
        lr = 0.05 / (1 + loss)
        ctx_norm = mx.linalg.norm(ctx)
        if float(ctx_norm) > 1e-6:
            ctx_n = ctx / (ctx_norm + 1e-8)
            new = old + lr * (ctx_n - old)
            new_norm = mx.linalg.norm(new)
            if float(new_norm) > 1e-6:
                self.signatures[cart.id] = new / (new_norm + 1e-8)
        
        self.hebbian(cart.id, loss < 3.0)


# =============================================================================
# GLOBAL WORKSPACE (Conscious Binding)
# =============================================================================

class Workspace:
    def __init__(self, d: int, slots: int = 4):
        self.d = d
        self.slots = slots
        self.state = mx.zeros((slots, d), dtype=mx.float16)
    
    def compete(self, outputs: List[mx.array], scores: List[float]) -> mx.array:
        if not outputs:
            return mx.zeros((self.d,), dtype=mx.float16)
        
        stacked = mx.stack([mx.mean(o, axis=(0, 1)) for o in outputs])
        k = min(self.slots, len(outputs))
        
        weights = mx.softmax(mx.array(scores[:k]))
        self.state = stacked[:k]
        
        broadcast = mx.sum(self.state * weights.reshape(-1, 1), axis=0)
        return broadcast


# =============================================================================
# TOKEN REGISTRY
# =============================================================================

class Registry:
    def __init__(self):
        self.tokens = ['<pad>', '<unk>']
        self.tok2id = {'<pad>': 0, '<unk>': 1}
        self.tok2cart = {}
        self.cooc = defaultdict(int)
        self.lock = threading.Lock()
    
    def add(self, text: str) -> List[int]:
        words = [w.lower() for w in text.split() if any(c.isalnum() for c in w)]
        if not words:
            return []
        
        with self.lock:
            ids = []
            for w in words:
                tid = self.tok2id.get(w)
                if tid is None:
                    tid = len(self.tokens)
                    self.tokens.append(w)
                    self.tok2id[w] = tid
                    self.tok2cart[tid] = None
                ids.append(tid)
            
            if len(ids) > 1 and len(self.cooc) < 50000:
                for i, t1 in enumerate(ids):
                    for j in range(max(0, i-5), min(len(ids), i+6)):
                        if i != j:
                            self.cooc[tuple(sorted((t1, ids[j])))] += 1
        return ids
    
    def unowned(self) -> List[int]:
        with self.lock:
            return [t for t, c in self.tok2cart.items() if c is None]
    
    def assign(self, tids: List[int], cid: str):
        with self.lock:
            for t in tids:
                self.tok2cart[t] = cid
    
    def cluster(self, seeds: List[int], maxsize: int = None) -> List[int]:
        # FIX: Use smaller max size by default
        maxsize = maxsize or CFG.cart_vocab
        with self.lock:
            scores = defaultdict(int)
            seedset = set(seeds)
            for (t1, t2), cnt in self.cooc.items():
                if t1 in seedset:
                    scores[t2] += cnt
                if t2 in seedset:
                    scores[t1] += cnt
            
            cluster = list(seedset)
            for tid, _ in sorted(scores.items(), key=lambda x: -x[1]):
                if tid not in seedset and len(cluster) < maxsize:
                    cluster.append(tid)
            return cluster[:maxsize]
    
    def save(self, path: str):
        with self.lock:
            data = {'tokens': self.tokens, 'tok2cart': {str(k): v for k, v in self.tok2cart.items()}}
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def load(self, path: str):
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        with self.lock:
            self.tokens = data['tokens']
            self.tok2id = {t: i for i, t in enumerate(self.tokens)}
            self.tok2cart = {int(k): v for k, v in data.get('tok2cart', {}).items()}


# =============================================================================
# ENGINE
# =============================================================================

class Engine:
    def __init__(self):
        log.info("Initializing engine...")
        
        self.stem = Stem(vocab=32000, d=CFG.dim, depth=CFG.stem_depth, heads=CFG.heads)
        self.stem.set_dtype(mx.float16)
        mx.eval(self.stem.parameters())
        log.info("Stem initialized")
        
        self.registry = Registry()
        self.carts: Dict[str, Cartridge] = {}
        self.router = Router(CFG.dim)
        self.workspace = Workspace(CFG.dim, CFG.workspace_slots)
        
        # Round-robin state
        self.cart_order: List[str] = []
        self.current_cart_idx: int = 0
        self.steps_on_current: int = 0
        
        # Start Gutenberg downloader
        global GUTENBERG_DOWNLOADER
        if GUTENBERG_DOWNLOADER is None and CFG.gutenberg_enabled:
            GUTENBERG_DOWNLOADER = GutenbergDownloader(
                download_dir='./training_data/gutenberg',
                max_books=CFG.gutenberg_max_books,
                delay=CFG.gutenberg_download_delay
            )
            GUTENBERG_DOWNLOADER.start()
        
        self._load()
        log.info(f"Engine ready: {len(self.carts)} cartridges, {len(self.registry.tokens)} tokens")
    
    def _load(self):
        reg_path = os.path.join(CFG.library, 'registry.json')
        if os.path.exists(reg_path):
            self.registry.load(reg_path)
        
        if os.path.exists(CFG.library):
            total_steps = 0  # Track total loaded steps
            for uid in os.listdir(CFG.library):
                p = os.path.join(CFG.library, uid)
                if os.path.isdir(p):
                    cart = Cartridge.load(uid, self.stem, CFG.library)
                    if cart:
                        self.carts[cart.id] = cart
                        self.router.register(cart)
                        self.cart_order.append(cart.id)
                        total_steps += cart.steps
            
            log.info(f"📦 Loaded {len(self.carts)} cartridges with {total_steps:,} total training steps")
    
    def spawn(self, tokens: List[int]) -> Cartridge:
        cart = Cartridge(self.stem, set(tokens))
        self.carts[cart.id] = cart
        self.registry.assign(tokens, cart.id)
        self.router.register(cart)
        self.cart_order.append(cart.id)
        log.info(f"Spawned {cart.id} with {len(tokens)} tokens (total: {len(self.carts)} cartridges)")
        return cart
    
    def context_emb(self, tokens: List[int]) -> mx.array:
        if not tokens:
            return mx.zeros((CFG.dim,), dtype=mx.float16)
        x = mx.array([tokens[:CFG.seq_len]])
        h = self.stem(x)
        return mx.mean(h, axis=(0, 1))
    
    def get_next_cartridge(self) -> Optional[Cartridge]:
        """Round-robin cartridge selection for training diversity"""
        if not self.carts:
            return None
        
        # Update cart_order if needed
        current_ids = set(self.carts.keys())
        order_ids = set(self.cart_order)
        if current_ids != order_ids:
            self.cart_order = list(current_ids)
            self.current_cart_idx = 0
        
        if not self.cart_order:
            return None
        
        self.current_cart_idx = self.current_cart_idx % len(self.cart_order)
        cart_id = self.cart_order[self.current_cart_idx]
        return self.carts.get(cart_id)
    
    def advance_round_robin(self):
        """Move to next cartridge in rotation"""
        self.steps_on_current += 1
        if self.steps_on_current >= CFG.round_robin_interval:
            self.current_cart_idx = (self.current_cart_idx + 1) % max(1, len(self.cart_order))
            self.steps_on_current = 0
    
    def train_step(self, batch: np.ndarray) -> Tuple[float, str]:
        x, y = batch[:, :-1], batch[:, 1:]
        ctx = self.context_emb(x[0].tolist())
        
        # HYBRID: Use round-robin with routing influence
        if not self.carts:
            # Bootstrap first cartridge
            unowned = self.registry.unowned()
            if unowned:
                # FIX: Limit initial cluster size
                cluster = self.registry.cluster(unowned[:30], maxsize=CFG.cart_vocab)
                cart = self.spawn(cluster)
            else:
                cart = self.spawn(list(range(2, min(50, len(self.registry.tokens)))))
        else:
            # Get cartridge via round-robin (ensures all get trained)
            cart = self.get_next_cartridge()
            if not cart:
                cart = list(self.carts.values())[0]
            
            # Also check routing - if routed cart has much lower loss, use it instead
            routed = self.router.route(ctx, k=1)
            if routed:
                routed_cart = self.carts.get(routed[0][0])
                if routed_cart and routed_cart.avg_loss < cart.avg_loss * 0.8:
                    cart = routed_cart
        
        # Ensure cartridge has tokens for this batch
        batch_tokens = set(np.unique(np.concatenate([x.flatten(), y.flatten()])))
        cart.add_tokens(batch_tokens)
        
        loss = cart.train_step(x, y)
        self.router.update_sig(cart, ctx, loss)
        self.advance_round_robin()
        
        return loss, cart.id
    
    def maybe_spawn(self, tokens: np.ndarray, step: int):
        if step % CFG.spawn_interval != 0:
            return
        
        unowned = []
        for t in np.unique(tokens):
            if self.registry.tok2cart.get(int(t)) is None:
                unowned.append(int(t))
        
        # FIX: Lower spawn threshold
        if len(unowned) >= CFG.spawn_min_tokens:
            cluster = self.registry.cluster(unowned, maxsize=CFG.cart_vocab)
            if len(cluster) >= CFG.spawn_min_tokens:
                self.spawn(cluster)
    
    def generate(self, prompt: str, max_tokens: int = 30) -> str:
        tokens = self.registry.add(prompt)
        if not tokens:
            return "[Empty prompt]"
        if not self.carts:
            return "[No cartridges available]"
        
        ctx = self.context_emb(tokens)
        
        # Try routing first
        routed = self.router.route(ctx, k=CFG.active_k)
        
        if routed:
            # Use multiple cartridges, combine outputs
            all_outputs = []
            scores = []
            
            for cart_id, score in routed:
                cart = self.carts.get(cart_id)
                if cart:
                    out_ids = cart.generate(tokens, max_new=max_tokens)
                    all_outputs.append(out_ids)
                    scores.append(score)
            
            if all_outputs:
                # Use output from highest-scoring cartridge
                best_idx = np.argmax(scores)
                out_ids = all_outputs[best_idx]
            else:
                out_ids = []
        else:
            # Fallback to first cartridge
            cart = list(self.carts.values())[0]
            out_ids = cart.generate(tokens, max_new=max_tokens)
        
        words = [self.registry.tokens[t] if t < len(self.registry.tokens) else '<unk>' for t in out_ids]
        return ' '.join(words) if words else "[No output generated]"
    
    def save(self):
        """Thread-safe save with full GPU synchronization"""
        self.registry.save(os.path.join(CFG.library, 'registry.json'))
        
        # Force all pending GPU operations to complete
        mx.eval()
        
        with GPU_LOCK:
            for cart in self.carts.values():
                cart.save(CFG.library)
        
        # Final sync to ensure writes complete
        mx.eval()
        log.info(f"💾 Saved {len(self.carts)} cartridges")


# =============================================================================
# GLOBAL STATE
# =============================================================================

@dataclass
class State:
    step: int = 0
    loss: float = 0.0
    active_cart: str = 'Init'
    tok_per_sec: float = 0.0
    num_carts: int = 0
    vocab_size: int = 0
    batch_size: int = 0
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    cpu_percent: float = 0.0
    paused: bool = False
    mode: str = 'gpu'
    chat_q: queue.Queue = field(default_factory=queue.Queue)
    resp_q: queue.Queue = field(default_factory=queue.Queue)
    lock: threading.Lock = field(default_factory=threading.Lock)
    
    # Track which cartridges have been trained recently
    recent_carts: List[str] = field(default_factory=list)
    
    def metrics(self) -> dict:
        with self.lock:
            return {
                'step': self.step, 
                'loss': self.loss, 
                'active_cartridge': self.active_cart,
                'tok_per_sec': self.tok_per_sec, 
                'num_cartridges': self.num_carts,
                'vocab_size': self.vocab_size, 
                'batch_size': self.batch_size,
                'memory_mb': self.memory_mb, 
                'memory_percent': self.memory_percent,
                'cpu_percent': self.cpu_percent, 
                'is_paused': self.paused,
                'training_mode': self.mode,
                'recent_carts': list(self.recent_carts[-5:])  # Last 5 trained
            }
    
    def save(self):
        """Persist training state to disk"""
        with self.lock:
            data = {
                'step': self.step,
                'paused': self.paused,
                'mode': self.mode,
                'last_save': time.time()
            }
        try:
            tmp_path = CFG.state_file + '.tmp'
            with open(tmp_path, 'w') as f:
                json.dump(data, f)
            Path(tmp_path).replace(CFG.state_file)  # Atomic write
            log.debug(f"💾 State saved: step {self.step}")
        except Exception as e:
            log.error(f"State save failed: {e}")
    
    def load(self):
        """Restore training state from disk"""
        if not os.path.exists(CFG.state_file):
            log.info("ℹ️  No saved state found, starting fresh")
            return
        
        try:
            with open(CFG.state_file) as f:
                data = json.load(f)
            with self.lock:
                self.step = data.get('step', 0)
                self.paused = data.get('paused', False)
                self.mode = data.get('mode', 'gpu')
            log.info(f"📂 State loaded: step {self.step}, mode {self.mode}")
        except Exception as e:
            log.warning(f"State load failed: {e}, using defaults")

STATE = State()
ENGINE: Optional[Engine] = None
GUTENBERG_DOWNLOADER = None


# =============================================================================
# GUTENBERG DOWNLOADER
# =============================================================================

class GutenbergDownloader:
    """Background downloader for Project Gutenberg books"""
    
    def __init__(self, download_dir: str, max_books: int = 1000, delay: int = 60):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.max_books = max_books
        self.delay = delay
        self.downloaded = self._load_downloaded()
        self.lock = threading.Lock()
        self.active = True
        
    def _load_downloaded(self) -> set:
        """Load set of already downloaded book IDs"""
        index_file = self.download_dir / 'downloaded.json'
        if index_file.exists():
            try:
                with open(index_file) as f:
                    return set(json.load(f))
            except:
                pass
        return set()
    
    def _save_downloaded(self):
        """Persist downloaded book IDs atomically"""
        index_file = self.download_dir / 'downloaded.json'
        tmp_file = index_file.with_suffix('.tmp')
        with open(tmp_file, 'w') as f:
            json.dump(list(self.downloaded), f)
        tmp_file.replace(index_file)
    
    def _extract_content(self, text: str) -> str:
        """Extract actual book content, remove Gutenberg boilerplate"""
        # Look for standard markers
        start_match = re.search(r'\*\*\* START OF.*?\*\*\*', text, re.IGNORECASE | re.DOTALL)
        end_match = re.search(r'\*\*\* END OF.*?\*\*\*', text, re.IGNORECASE | re.DOTALL)
        
        if start_match and end_match:
            content = text[start_match.end():end_match.start()]
        else:
            # Fallback: remove common header/footer patterns
            lines = text.split('\n')
            content_lines = []
            in_content = False
            
            for line in lines:
                if '*** START OF' in line or 'START OF THE PROJECT' in line:
                    in_content = True
                    continue
                if '*** END OF' in line or 'END OF THE PROJECT' in line:
                    break
                if in_content:
                    content_lines.append(line)
            
            content = '\n'.join(content_lines)
        
        # Clean up excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        return content.strip()
    
    def _download_book(self, book_id: int) -> bool:
        """Download a single book by ID"""
        try:
            # Gutenberg mirror URL pattern
            url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
            
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                log.debug(f"Book {book_id} not found (status {response.status_code})")
                return False
            
            content = self._extract_content(response.text)
            
            if len(content) < CFG.gutenberg_min_length:
                log.debug(f"Book {book_id} too short ({len(content)} chars), skipping")
                return False
            
            # Clean filename
            filename = self.download_dir / f"gutenberg_{book_id:05d}.txt"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            
            with self.lock:
                self.downloaded.add(book_id)
                self._save_downloaded()
            
            log.info(f"📚 Downloaded book {book_id} ({len(content):,} chars)")
            return True
            
        except Exception as e:
            log.warning(f"Failed to download book {book_id}: {e}")
            return False
    
    def download_worker(self):
        """Background worker thread - polite downloader"""
        log.info(f"📖 Gutenberg downloader started (delay: {self.delay}s, max: {self.max_books})")
        
        while self.active and len(self.downloaded) < self.max_books:
            try:
                # Strategy: Random books in popular range (1000-70000)
                book_id = random.randint(1000, 70000)
                
                if book_id in self.downloaded:
                    continue
                
                success = self._download_book(book_id)
                
                if success:
                    time.sleep(self.delay)  # Be polite
                else:
                    time.sleep(5)  # Wait before trying another ID
                    
            except Exception as e:
                log.error(f"Downloader worker error: {e}")
                time.sleep(10)
        
        log.info(f"📚 Downloader finished. Total books: {len(self.downloaded)}")
    
    def start(self):
        """Start background download thread"""
        if not CFG.gutenberg_enabled:
            log.info("📚 Gutenberg downloader disabled")
            return
        
        thread = threading.Thread(target=self.download_worker, daemon=True, name="GutenbergDownloader")
        thread.start()
        log.info("🚀 Gutenberg downloader thread started")


# =============================================================================
# DATA GENERATOR
# =============================================================================

def data_gen():
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models require training data to learn patterns.",
        "Neural networks approximate complex mathematical functions.",
        "Attention mechanisms help models focus on relevant information.",
        "Transformers have revolutionized natural language processing.",
        "Embeddings map discrete tokens to continuous vector spaces.",
        "Backpropagation computes gradients for optimization algorithms.",
        "Apple Silicon provides unified memory architecture for efficiency.",
        "The brain uses modular specialized regions for different tasks.",
        "Consciousness emerges from complex patterns of neural activity.",
        "The prefrontal cortex coordinates executive function and planning.",
        "Memory consolidation occurs during deep sleep phases.",
        "Synaptic plasticity enables learning and adaptation over time.",
        "Visual processing begins in the primary visual cortex area.",
        "Language comprehension involves multiple distributed brain regions.",
        "The hippocampus plays a crucial role in memory formation.",
        "Dopamine pathways are involved in reward and motivation.",
        "The cerebellum coordinates fine motor movements precisely.",
        "Neurons communicate through electrical and chemical signals.",
        "The autonomic nervous system regulates involuntary functions.",
    ]
    
    base_dir = Path('training_data')
    gutenberg_dir = base_dir / 'gutenberg'
    base_dir.mkdir(exist_ok=True)
    gutenberg_dir.mkdir(exist_ok=True)
    
    fidx = 0
    while True:
        # Build fresh file list periodically (to catch new downloads)
        files = []
        if base_dir.exists():
            files.extend([base_dir / f for f in os.listdir(base_dir) if f.endswith('.txt') and not f.startswith('gutenberg_')])
        
        # Include gutenberg files
        if gutenberg_dir.exists():
            files.extend([gutenberg_dir / f for f in os.listdir(gutenberg_dir) if f.endswith('.txt')])
        
        if files:
            try:
                file_path = files[fidx % len(files)]
                with open(file_path, 'r', errors='ignore') as f:
                    text = f.read()
                
                if len(text) > 100:
                    # Yield chunks to avoid overwhelming memory
                    for i in range(0, len(text), 3000):
                        yield text[i:i+3000]
                else:
                    yield ' '.join(random.choices(sentences, k=random.randint(10, 30)))
            except Exception as e:
                log.debug(f"Error reading file {file_path}: {e}")
                yield ' '.join(random.choices(sentences, k=random.randint(10, 30)))
            fidx += 1
        else:
            # No files yet, use sentences
            yield ' '.join(random.choices(sentences, k=random.randint(10, 30)))


# =============================================================================
# TRAINING LOOP
# =============================================================================

def training_loop():
    global ENGINE, TUNER
    log.info("Training loop started")
    
    gen = data_gen()
    buf = []
    step = 0
    tokens_done = 0
    t0 = time.time()
    step_start = time.time()
    consecutive_errors = 0
    MAX_ERRORS = 5
    
    while True:
        try:
            if STATE.paused:
                time.sleep(0.05)
                continue
            
            batch_size = TUNER.batch
            target = batch_size * (CFG.seq_len + 1) * 2
            
            # Fill buffer (no GPU needed)
            while len(buf) < target:
                text = next(gen)
                toks = ENGINE.registry.add(text)
                buf.extend(toks)
            
            if len(buf) < batch_size * (CFG.seq_len + 1):
                time.sleep(0.01)
                continue
            
            # Build batch
            batch = []
            idx = 0
            for _ in range(batch_size):
                end = idx + CFG.seq_len + 1
                if end > len(buf):
                    break
                batch.append(buf[idx:end])
                idx += CFG.seq_len
            buf = buf[idx:]
            
            if len(batch) < batch_size:
                continue
            
            np_batch = np.array(batch, dtype=np.int32)
            
            # GPU work with lock
            with GPU_LOCK:
                ENGINE.maybe_spawn(np_batch.flatten(), step)
                loss, cid = ENGINE.train_step(np_batch)
                mx.eval()  # Force sync inside lock
            
            # Validate loss
            if not np.isfinite(loss):
                consecutive_errors += 1
                if consecutive_errors > 5:
                    TUNER.on_error()
                    consecutive_errors = 0
                continue
            
            consecutive_errors = 0
            
            # Update tuner
            step_time = time.time() - step_start
            TUNER.update(step_time)
            step_start = time.time()
            
            step += 1
            tokens_done += batch_size * CFG.seq_len
            
            # Update state
            with STATE.lock:
                STATE.active_cart = cid
                STATE.batch_size = TUNER.batch
                if cid not in STATE.recent_carts:
                    STATE.recent_carts.append(cid)
                    if len(STATE.recent_carts) > 20:
                        STATE.recent_carts.pop(0)
            
            # Periodic updates
            if step % 10 == 0:
                now = time.time()
                elapsed = now - t0
                tps = tokens_done / elapsed if elapsed > 0 else 0
                
                mem_mb, mem_pct, cpu_pct = 0.0, 0.0, 0.0
                if HAS_PSUTIL:
                    try:
                        proc = psutil.Process()
                        mem_mb = proc.memory_info().rss / (1024 * 1024)
                        mem_pct = proc.memory_percent()
                        cpu_pct = psutil.cpu_percent()
                    except:
                        pass
                
                with STATE.lock:
                    STATE.step = step
                    STATE.loss = float(loss)
                    STATE.num_carts = len(ENGINE.carts)
                    STATE.tok_per_sec = tps
                    STATE.vocab_size = len(ENGINE.registry.tokens)
                    STATE.memory_mb = mem_mb
                    STATE.memory_percent = mem_pct
                    STATE.cpu_percent = cpu_pct
                
                if step % 50 == 0:
                    tokens_done = 0
                    t0 = now
            
            # Periodic state save
            if step % 500 == 0:
                try:
                    STATE.save()
                    TUNER.save()  # Save tuner state periodically
                except Exception as e:
                    log.error(f"Periodic state save error: {e}")
            
            if step % CFG.save_interval == 0:
                try:
                    ENGINE.save()
                except Exception as e:
                    log.error(f"Save error: {e}")
        
        except KeyboardInterrupt:
            log.info("Training interrupted")
            ENGINE.save()
            break
        except Exception as e:
            log.error(f"Training error: {e}")
            traceback.print_exc()
            consecutive_errors += 1
            
            if consecutive_errors >= MAX_ERRORS:
                log.critical("Too many errors, pausing training")
                STATE.paused = True
                if ENGINE:
                    try:
                        ENGINE.save()
                    except Exception as save_err:
                        log.error(f"Save failed during error recovery: {save_err}")
                break
            
            # Reduce batch on any error
            TUNER.on_error()
            time.sleep(1.0)  # Longer delay


def chat_loop():
    global ENGINE
    log.info("Chat loop started")
    
    while True:
        try:
            if not STATE.chat_q.empty():
                try:
                    req = STATE.chat_q.get_nowait()
                except:
                    time.sleep(0.02)
                    continue
                
                try:
                    prompt = req.get('prompt', '')
                    conv = req.get('conversation', [])
                    
                    if conv:
                        parts = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in conv]
                        context = ' '.join(parts[-3:])
                    else:
                        context = prompt
                    
                    if not context.strip():
                        STATE.resp_q.put({'error': 'Empty prompt'})
                    else:
                        max_tokens = req.get('max_tokens', 50)
                        with GPU_LOCK:
                            response = ENGINE.generate(context, max_tokens=max_tokens)
                        STATE.resp_q.put({
                            'text': response,
                            'cartridge_count': len(ENGINE.carts),
                            'vocab_size': len(ENGINE.registry.tokens)
                        })
                except Exception as e:
                    log.error(f"Generation error: {e}")
                    STATE.resp_q.put({'error': str(e)})
            else:
                time.sleep(0.02)
        except Exception as e:
            log.error(f"Chat loop error: {e}")
            time.sleep(0.1)


# =============================================================================
# FASTAPI SERVER
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup - Validate dependencies
    log.info("🔍 Validating dependencies...")
    log.info(f"✅ WebSockets: {websockets.__version__}")
    log.info(f"✅ SSE Starlette: {sse_starlette.__version__}")
    
    # Load state before anything else
    STATE.load()
    
    # Load BatchTuner state (must happen before engine initialization)
    TUNER.load()
    
    # Set device mode from saved state
    mx.set_default_device(mx.cpu if STATE.mode == 'cpu' else mx.gpu)
    
    # Initialize engine
    global ENGINE
    ENGINE = Engine()
    
    training_thread = threading.Thread(target=training_loop, daemon=True, name="TrainingLoop")
    chat_thread = threading.Thread(target=chat_loop, daemon=True, name="ChatLoop")
    training_thread.start()
    chat_thread.start()
    
    log.info("✅ Server started on http://localhost:8000")
    log.info("📡 WebSocket endpoint: ws://localhost:8000/ws/chat")
    yield
    
    # Shutdown
    log.info("🛑 Shutting down...")
    STATE.paused = True
    
    # Wait for current GPU operations to finish
    with GPU_LOCK:
        time.sleep(0.5)  # Let GPU operations complete
    
    # Save everything
    STATE.save()  # Save training state
    TUNER.save()  # Save tuner state
    if ENGINE:
        ENGINE.save()  # Save cartridges
    
    log.info("💾 Final save complete")

app = FastAPI(title="Infinite Cartridge System", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
async def root():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"msg": "Infinite Cartridge System - No index.html found"}


@app.get("/stream/metrics")
async def stream_metrics(request: Request):
    async def gen():
        while True:
            if await request.is_disconnected():
                break
            yield {"data": json.dumps(STATE.metrics())}
            await asyncio.sleep(0.5)
    return EventSourceResponse(gen())


@app.get("/cartridges")
def list_carts(detailed: bool = False):
    if not ENGINE:
        return {"summary": {"total_cartridges": 0, "total_owned_tokens": 0}, "cartridges": []}
    
    try:
        # Copy to avoid iteration error
        carts_snapshot = list(ENGINE.carts.values())
    except RuntimeError:
        return {"summary": {"total_cartridges": 0, "total_owned_tokens": 0}, "cartridges": [], "error": "retry"}
    
    carts = []
    for c in carts_snapshot:
        try:
            avg_loss = None
            if c.loss_count > 0:
                loss_val = c.avg_loss
                if np.isfinite(loss_val):
                    avg_loss = round(loss_val, 3)
            
            carts.append({
                "id": c.id, 
                "tokens": len(c.tokens), 
                "created": c.created, 
                "train_steps": c.steps, 
                "strength": c.strength,
                "avg_loss": avg_loss
            })
        except:
            pass
    
    carts.sort(key=lambda x: -x['train_steps'])
    
    return {
        "summary": {"total_cartridges": len(carts), "total_owned_tokens": sum(c['tokens'] for c in carts)},
        "cartridges": carts
    }


@app.post("/training/pause")
def pause():
    STATE.paused = True
    return {"status": "paused"}


@app.post("/training/resume")
def resume():
    STATE.paused = False
    return {"status": "resumed"}


@app.post("/training/mode/{mode}")
def set_mode(mode: str):
    if mode in ['cpu', 'gpu']:
        STATE.mode = mode
        mx.set_default_device(mx.cpu if mode == 'cpu' else mx.gpu)
        return {"mode": mode}
    return {"error": "Invalid mode"}


@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket):
    client_ip = websocket.client.host if websocket.client else "unknown"
    user_agent = websocket.headers.get("user-agent", "unknown")[:30]
    
    log.info(f"🌐 WebSocket ATTEMPT from {client_ip} | {user_agent}")
    
    try:
        await websocket.accept()
        log.info(f"✅ WebSocket CONNECTED from {client_ip}")
    except Exception as e:
        log.error(f"❌ WebSocket ACCEPT FAILED: {e}")
        return
    
    try:
        while True:
            data = await websocket.receive_json()
            prompt_preview = data.get('prompt', '')[:40]
            log.debug(f"💬 Message: '{prompt_preview}...'")
            
            # Clear stale responses
            while not STATE.resp_q.empty():
                try:
                    STATE.resp_q.get_nowait()
                except:
                    pass
            
            STATE.chat_q.put(data)
            
            # Wait for response
            t0 = time.time()
            response = None
            while time.time() - t0 < 30:
                try:
                    response = STATE.resp_q.get_nowait()
                    break
                except:
                    await asyncio.sleep(0.1)
            
            if response is None:
                response = {"error": "Timeout"}
                log.warning(f"⏰ Timeout for {client_ip}")
            
            await websocket.send_json(response)
            log.debug(f"📤 Response sent to {client_ip}")
            
    except WebSocketDisconnect:
        log.info(f"👋 DISCONNECTED from {client_ip}")
    except Exception as e:
        log.error(f"💥 ERROR from {client_ip}: {e}")
        try:
            await websocket.send_json({"error": str(e)})
        except:
            pass


@app.get("/metrics")
def get_metrics():
    return STATE.metrics()


@app.get("/tuner")
def get_tuner():
    return TUNER.status()


@app.get("/tuner/histogram")
def get_tuner_histogram():
    """Get TPS histogram data for UI visualization"""
    return TUNER.get_histogram(bins=10)


@app.get("/gutenberg/count")
def gutenberg_count():
    if not CFG.gutenberg_enabled:
        return {"total": 0, "enabled": False}
    
    gutenberg_dir = Path('training_data/gutenberg')
    if not gutenberg_dir.exists():
        return {"total": 0, "enabled": True}
    
    files = list(gutenberg_dir.glob("*.txt"))
    return {"total": len(files), "enabled": True}


@app.get("/debug")
def debug_info():
    """Debug endpoint for troubleshooting"""
    if not ENGINE:
        return {"error": "Engine not initialized"}
    
    try:
        carts_snapshot = list(ENGINE.carts.values())
    except RuntimeError:
        return {"error": "retry"}
    
    cart_info = []
    for c in carts_snapshot:
        try:
            avg_loss = None
            if c.loss_count > 0:
                loss_val = c.avg_loss
                if np.isfinite(loss_val):
                    avg_loss = round(loss_val, 3)
            
            signal_norm = float(mx.linalg.norm(c.signal))
            if not np.isfinite(signal_norm):
                signal_norm = 0.0
            
            cart_info.append({
                "id": c.id,
                "tokens": len(c.tokens),
                "steps": c.steps,
                "avg_loss": avg_loss,
                "signal_norm": signal_norm
            })
        except:
            pass
    
    return {
        "cartridges": cart_info,
        "vocab_size": len(ENGINE.registry.tokens),
        "cart_order": list(ENGINE.cart_order) if hasattr(ENGINE, 'cart_order') else [],
        "current_idx": ENGINE.current_cart_idx if hasattr(ENGINE, 'current_cart_idx') else 0,
        "router_signatures": len(ENGINE.router.signatures)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
