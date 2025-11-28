#!/usr/bin/env python3
"""
Infinite Cartridge System - Brain-Inspired Modular Neural Network
=================================================================
Millions of tiny specialized "cartridges" (~100KB) share frozen universal "stem" layers.
Learned router (prefrontal cortex) selects 2-4 experts per query.
Global workspace binds outputs. Hebbian learning strengthens successful pathways.

Optimized for Apple Silicon: cartridges fit in L2 cache for streaming inference.
"""

import os, uuid, json, time, queue, asyncio, logging, threading, traceback, random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path

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
    spawn_interval: int = 50        # More frequent spawn checks
    spawn_min_tokens: int = 30      # Lower threshold for spawning
    save_interval: int = 2000
    
    # Round-robin training to ensure all cartridges get trained
    round_robin_interval: int = 20  # Switch cartridge every N steps

CFG = Config()
os.makedirs(CFG.library, exist_ok=True)


class BatchTuner:
    """PID-style batch optimizer - finds max throughput"""
    def __init__(self, initial: int = 8, min_b: int = 8, max_b: int = 512):
        self.batch = initial
        self.min_b = min_b
        self.max_b = max_b
        self.best_tps = 0.0
        self.best_batch = initial
        self.explore_cooldown = 0
        self.direction = 1

    def update(self, step_time: float) -> int:
        tps = (self.batch * CFG.seq_len) / max(step_time, 0.001)
        
        if tps > self.best_tps * 1.02:
            self.best_tps = tps
            self.best_batch = self.batch
            self.explore_cooldown = 0
        else:
            self.explore_cooldown += 1
        
        if self.explore_cooldown < 15:
            if self.direction == 1:
                self.batch = min(self.max_b, int(self.batch * 1.3))
            else:
                self.batch = max(self.min_b, int(self.batch * 0.7))
            if self.batch >= self.max_b or self.batch <= self.min_b:
                self.direction *= -1
        else:
            self.batch = self.best_batch
        
        return self.batch

    def status(self) -> dict:
        return {"current": self.batch, "best": self.best_batch, "best_tps": round(self.best_tps, 1)}


TUNER = BatchTuner(initial=8, min_b=8, max_b=512)

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
                 'strength', 'total_loss', 'loss_count')
    
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
        """Expand vocabulary with new tokens"""
        added = new_tokens - self.tokens
        if not added:
            return
        
        self.tokens.update(added)
        old_vocab_size = self.vocab_size
        
        # Rebuild mapping
        self.local_to_global = [0, 1] + sorted(self.tokens)
        self.global_to_local = {g: i for i, g in enumerate(self.local_to_global)}
        self.vocab_size = len(self.local_to_global)
        
        # Extend lookup
        maxg = max(self.local_to_global)
        if maxg >= len(self._lookup):
            new_lookup = np.ones(maxg + 1000, dtype=np.int32)
            new_lookup[:len(self._lookup)] = self._lookup
            self._lookup = new_lookup
        for g, l in self.global_to_local.items():
            self._lookup[g] = l
        
        # Expand head if needed
        if self.vocab_size > old_vocab_size:
            old_weight = self.head.weight
            new_weight = mx.zeros((self.vocab_size, CFG.dim), dtype=mx.float16)
            new_weight[:old_vocab_size, :] = old_weight
            # Initialize new weights with small random values
            new_weight[old_vocab_size:, :] = mx.random.normal(
                (self.vocab_size - old_vocab_size, CFG.dim), dtype=mx.float16
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
        
        params = {'signal': self.signal}
        for k, v in tree_flatten(self.lora.parameters()):
            params[f'lora.{k}'] = v
        for k, v in tree_flatten(self.head.parameters()):
            params[f'head.{k}'] = v
        mx.save_safetensors(str(folder / 'weights.safetensors'), params)
        
        with open(folder / 'meta.json', 'w') as f:
            json.dump({
                'id': self.id, 
                'tokens': list(self.tokens), 
                'created': self.created,
                'steps': self.steps, 
                'strength': self.strength,
                'total_loss': self.total_loss,
                'loss_count': self.loss_count
            }, f)
    
    @classmethod
    def load(cls, uid: str, stem: Stem, path: str) -> Optional['Cartridge']:
        folder = Path(path) / uid
        if not folder.exists():
            return None
        try:
            with open(folder / 'meta.json') as f:
                meta = json.load(f)
            cart = cls(stem, set(meta['tokens']), uid)
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
                # Load lora and head weights too
                lora_params = {k.replace('lora.', ''): v for k, v in W.items() if k.startswith('lora.')}
                head_params = {k.replace('head.', ''): v for k, v in W.items() if k.startswith('head.')}
                if lora_params:
                    for (name, _), (_, val) in zip(tree_flatten(cart.lora.parameters()), lora_params.items()):
                        pass  # Would need proper update logic
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
        
        self._load()
        log.info(f"Engine ready: {len(self.carts)} cartridges, {len(self.registry.tokens)} tokens")
    
    def _load(self):
        reg_path = os.path.join(CFG.library, 'registry.json')
        if os.path.exists(reg_path):
            self.registry.load(reg_path)
        
        if os.path.exists(CFG.library):
            for uid in os.listdir(CFG.library):
                p = os.path.join(CFG.library, uid)
                if os.path.isdir(p):
                    cart = Cartridge.load(uid, self.stem, CFG.library)
                    if cart:
                        self.carts[cart.id] = cart
                        self.router.register(cart)
                        self.cart_order.append(cart.id)
    
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
        self.registry.save(os.path.join(CFG.library, 'registry.json'))
        for cart in self.carts.values():
            cart.save(CFG.library)
        log.info(f"Saved {len(self.carts)} cartridges")


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

STATE = State()
ENGINE: Optional[Engine] = None


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
    
    files = []
    if os.path.exists('training_data'):
        files = [os.path.join('training_data', f) for f in os.listdir('training_data') 
                 if f.endswith('.txt')]
    
    fidx = 0
    while True:
        if files:
            try:
                with open(files[fidx % len(files)], 'r', errors='ignore') as f:
                    text = f.read()
                if len(text) > 100:
                    for i in range(0, len(text), 3000):
                        yield text[i:i+3000]
                else:
                    yield ' '.join(random.choices(sentences, k=random.randint(10, 30)))
            except Exception as e:
                log.debug(f"Error reading file: {e}")
                yield ' '.join(random.choices(sentences, k=random.randint(10, 30)))
            fidx += 1
        else:
            yield ' '.join(random.choices(sentences, k=random.randint(10, 30)))


# =============================================================================
# TRAINING LOOP
# =============================================================================

def training_loop():
    global ENGINE
    log.info("Training loop started")
    
    gen = data_gen()
    buf = []
    step = 0
    tokens_done = 0
    t0 = time.time()
    step_start = time.time()
    
    while True:
        try:
            if STATE.paused:
                time.sleep(0.05)
                continue
            
            batch_size = TUNER.batch
            target = batch_size * (CFG.seq_len + 1) * 2
            while len(buf) < target:
                text = next(gen)
                toks = ENGINE.registry.add(text)
                buf.extend(toks)
            
            if len(buf) < batch_size * (CFG.seq_len + 1):
                time.sleep(0.01)
                continue
            
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
            
            ENGINE.maybe_spawn(np_batch.flatten(), step)
            loss, cid = ENGINE.train_step(np_batch)
            
            if not np.isfinite(loss):
                continue
            
            step_time = time.time() - step_start
            TUNER.update(step_time)
            step_start = time.time()
            
            step += 1
            tokens_done += batch_size * CFG.seq_len
            
            # Track recent cartridges
            with STATE.lock:
                STATE.active_cart = cid
                if cid not in STATE.recent_carts:
                    STATE.recent_carts.append(cid)
                    if len(STATE.recent_carts) > 20:
                        STATE.recent_carts.pop(0)
            
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
                    STATE.batch_size = TUNER.batch
                    STATE.memory_mb = mem_mb
                    STATE.memory_percent = mem_pct
                    STATE.cpu_percent = cpu_pct
                
                if step % 50 == 0:
                    tokens_done = 0
                    t0 = now
            
            if step % CFG.save_interval == 0:
                ENGINE.save()
        
        except KeyboardInterrupt:
            log.info("Training interrupted")
            ENGINE.save()
            break
        except Exception as e:
            log.error(f"Training error: {e}")
            traceback.print_exc()
            time.sleep(1)


def chat_loop():
    global ENGINE
    log.info("Chat loop started")
    
    while True:
        try:
            if not STATE.chat_q.empty():
                try:
                    req = STATE.chat_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                was_paused = STATE.paused
                STATE.paused = True
                time.sleep(0.1)
                
                try:
                    prompt = req.get('prompt', '')
                    conv = req.get('conversation', [])
                    
                    if conv:
                        parts = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in conv]
                        context = ' '.join(parts[-3:])  # Last 3 messages for context
                    else:
                        context = prompt
                    
                    if not context.strip():
                        STATE.resp_q.put({'error': 'Empty prompt'})
                    else:
                        max_tokens = req.get('max_tokens', 50)
                        response = ENGINE.generate(context, max_tokens=max_tokens)
                        STATE.resp_q.put({
                            'text': response,
                            'cartridge_count': len(ENGINE.carts),
                            'vocab_size': len(ENGINE.registry.tokens)
                        })
                except Exception as e:
                    log.error(f"Generation error: {e}")
                    traceback.print_exc()
                    STATE.resp_q.put({'error': str(e)})
                finally:
                    STATE.paused = was_paused
            else:
                time.sleep(0.02)
        except Exception as e:
            log.error(f"Chat loop error: {e}")
            time.sleep(0.1)


# =============================================================================
# FASTAPI SERVER
# =============================================================================

app = FastAPI(title="Infinite Cartridge System")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def startup():
    global ENGINE
    ENGINE = Engine()
    threading.Thread(target=training_loop, daemon=True, name="TrainingLoop").start()
    threading.Thread(target=chat_loop, daemon=True, name="ChatLoop").start()
    log.info("Server started on http://localhost:8000")


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
    try:
        await websocket.accept()
        log.info("WebSocket connected")
    except Exception as e:
        log.error(f"WebSocket accept failed: {e}")
        return
    try:
        while True:
            data = await websocket.receive_json()
            
            # Clear stale responses
            while not STATE.resp_q.empty():
                try:
                    STATE.resp_q.get_nowait()
                except:
                    pass
            
            STATE.chat_q.put(data)
            
            # NON-BLOCKING wait with async sleep
            t0 = time.time()
            response = None
            while time.time() - t0 < 30:
                try:
                    response = STATE.resp_q.get_nowait()  # Non-blocking
                    break
                except:
                    pass
                await asyncio.sleep(0.1)  # Yield to event loop
            
            if response is None:
                response = {"error": "Timeout waiting for response"}
            
            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        log.info("WebSocket disconnected")
    except Exception as e:
        log.error(f"WebSocket error: {e}")
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
