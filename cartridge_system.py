#!/usr/bin/env python3
"""
Infinite Cartridge System - Brain-Inspired Modular Neural Network
=================================================================
Millions of tiny specialized "cartridges" (~100KB) share frozen universal "stem" layers.
Learned router (prefrontal cortex) selects 2-4 experts per query.
Global workspace binds outputs. Hebbian learning strengthens successful pathways.

Optimized for Apple Silicon: cartridges fit in L2 cache for streaming inference.
"""

import os, uuid, json, time, queue, asyncio, logging, threading, traceback
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

# Optional system metrics
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
    dim: int = 256              # Model dimension
    stem_depth: int = 4         # Frozen backbone layers  
    heads: int = 8              # Attention heads
    adapter_rank: int = 16      # LoRA rank per cartridge
    
    cart_vocab: int = 1024      # Tokens per cartridge
    max_carts: int = 10000      # Scalable to millions
    active_k: int = 4           # Active cartridges per query
    workspace_slots: int = 4    # Global workspace capacity
    
    batch: int = 32
    seq_len: int = 64
    lr: float = 3e-4
    
    library: str = './cartridge_library'
    spawn_interval: int = 100
    spawn_min_tokens: int = 100
    save_interval: int = 2000

CFG = Config()
os.makedirs(CFG.library, exist_ok=True)

# =============================================================================
# NEURAL PRIMITIVES
# =============================================================================

class RMSNorm(nn.Module):
    """Fast RMS normalization"""
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.w = mx.ones((d,))
    
    def __call__(self, x):
        return x * mx.rsqrt(mx.mean(x*x, axis=-1, keepdims=True) + self.eps) * self.w


class SwiGLU(nn.Module):
    """SwiGLU MLP - fast and effective"""
    def __init__(self, d: int, mult: int = 4):
        super().__init__()
        h = d * mult
        self.w1 = nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(h, d, bias=False)  
        self.w3 = nn.Linear(d, h, bias=False)
    
    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    """Pre-norm transformer block"""
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
    """
    Universal processing layers - FROZEN after pretraining.
    All cartridges share this backbone. Like V1/A1 in cortex.
    ~5MB at fp16, loaded once, cached in SLC.
    """
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
    """Low-rank adapter - tiny trainable specialization"""
    def __init__(self, d: int, r: int = 16):
        super().__init__()
        self.A = nn.Linear(d, r, bias=False)
        self.B = nn.Linear(r, d, bias=False)
        # Initialize B to zero for identity at start
        self.B.weight = mx.zeros_like(self.B.weight)
    
    def __call__(self, x):
        return x + self.B(self.A(x)) * 0.1


class Cartridge:
    """
    Specialized expert (~100KB trainable params):
    - Shared STEM reference (frozen)
    - LoRA adapter (learns specialization)
    - Domain signal (identity vector)
    - Output head (project to local vocab)
    """
    __slots__ = ('id', 'stem', 'tokens', 'vocab_size', 'local_to_global', 'global_to_local',
                 '_lookup', 'lora', 'signal', 'head', 'created', 'last_used', 'steps', 'strength')
    
    def __init__(self, stem: Stem, tokens: set = None, uid: str = None):
        self.id = uid or uuid.uuid4().hex[:8]
        self.stem = stem
        self.tokens = set(tokens) if tokens else set()
        self.created = time.time()
        self.last_used = time.time()
        self.steps = 0
        self.strength = 1.0  # Hebbian connection strength
        
        # Build vocab mapping
        self.local_to_global = [0, 1] + sorted(self.tokens)  # 0=pad, 1=unk
        self.global_to_local = {g: i for i, g in enumerate(self.local_to_global)}
        self.vocab_size = len(self.local_to_global)
        
        # Vectorized lookup
        maxg = max(self.local_to_global) if self.local_to_global else 1
        self._lookup = np.ones(maxg + 1, dtype=np.int32)
        for g, l in self.global_to_local.items():
            self._lookup[g] = l
        
        # Trainable components (~100KB total at fp16)
        self.signal = mx.zeros((CFG.dim,), dtype=mx.float16)  # Domain identity
        self.lora = LoRA(CFG.dim, CFG.adapter_rank)
        self.head = nn.Linear(CFG.dim, self.vocab_size, bias=False)
        
        # Convert to fp16
        self.lora.set_dtype(mx.float16)
        self.head.set_dtype(mx.float16)
        mx.eval(self.lora.parameters(), self.head.parameters(), self.signal)
    
    def encode(self, global_ids) -> mx.array:
        """Global token IDs -> local IDs"""
        arr = np.asarray(global_ids, dtype=np.int32)
        mask = arr >= len(self._lookup)
        result = np.where(mask, 1, self._lookup[np.minimum(arr, len(self._lookup)-1)])
        return mx.array(result)
    
    def forward(self, x: mx.array, workspace: mx.array = None) -> Tuple[mx.array, mx.array]:
        """Forward: stem -> lora -> head"""
        self.last_used = time.time()
        h = self.stem(x, self.signal)
        h = self.lora(h)
        if workspace is not None:
            h = h + workspace.reshape(1, 1, -1) * 0.1
        return self.head(h), h
    
    def train_step(self, x_global: np.ndarray, y_global: np.ndarray) -> float:
        """Train on batch, return loss"""
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
        
        # Simple SGD update (fast)
        lr = CFG.lr
        for (_, p), (_, g) in zip(tree_flatten(self.lora.parameters()), tree_flatten(grads[0])):
            p -= lr * g
        for (_, p), (_, g) in zip(tree_flatten(self.head.parameters()), tree_flatten(grads[1])):
            p -= lr * g
        self.signal -= lr * grads[2]
        
        mx.eval(self.lora.parameters(), self.head.parameters(), self.signal)
        self.steps += 1
        return float(loss)
    
    def generate(self, tokens: List[int], max_new: int = 20, temp: float = 0.8) -> List[int]:
        """Autoregressive generation"""
        curr = list(tokens)
        out = []
        
        for _ in range(max_new):
            ctx = curr[-CFG.seq_len:]
            x = self.encode(ctx).reshape(1, -1)
            logits, _ = self.forward(x)
            mx.eval(logits)
            
            probs = mx.softmax(logits[0, -1] / temp)
            idx = int(mx.random.categorical(mx.log(probs + 1e-10)))
            
            if idx <= 1:  # pad/unk
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
            json.dump({'id': self.id, 'tokens': list(self.tokens), 'created': self.created,
                       'steps': self.steps, 'strength': self.strength}, f)
    
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
            
            wpath = folder / 'weights.safetensors'
            if wpath.exists():
                W = mx.load(str(wpath))
                if 'signal' in W:
                    cart.signal = W['signal']
            return cart
        except Exception as e:
            log.error(f"Load {uid} failed: {e}")
            return None


# =============================================================================
# ROUTER (Prefrontal Connector Hub)
# =============================================================================

class Router:
    """Routes queries to cartridges via learned signatures. Hebbian updates."""
    
    def __init__(self, d: int):
        self.d = d
        self.signatures: Dict[str, mx.array] = {}
        self.strengths: Dict[str, float] = {}
        self.encoder = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.encoder.set_dtype(mx.float16)
        mx.eval(self.encoder.parameters())
    
    def register(self, cart: Cartridge):
        self.signatures[cart.id] = cart.signal / (mx.linalg.norm(cart.signal) + 1e-8)
        self.strengths[cart.id] = cart.strength
    
    def route(self, ctx: mx.array, k: int = 4) -> List[Tuple[str, float]]:
        if not self.signatures:
            return []
        
        q = self.encoder(ctx)
        q = q / (mx.linalg.norm(q) + 1e-8)
        
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
        ctx_n = ctx / (mx.linalg.norm(ctx) + 1e-8)
        new = old + lr * (ctx_n - old)
        self.signatures[cart.id] = new / (mx.linalg.norm(new) + 1e-8)
        self.hebbian(cart.id, loss < 3.0)


# =============================================================================
# GLOBAL WORKSPACE (Conscious Binding)
# =============================================================================

class Workspace:
    """Top-k outputs compete for slots, winners broadcast to all"""
    
    def __init__(self, d: int, slots: int = 4):
        self.d = d
        self.slots = slots
        self.state = mx.zeros((slots, d), dtype=mx.float16)
    
    def compete(self, outputs: List[mx.array], scores: List[float]) -> mx.array:
        if not outputs:
            return mx.zeros((self.d,), dtype=mx.float16)
        
        # Stack and select top-k by score
        stacked = mx.stack([mx.mean(o, axis=(0, 1)) for o in outputs])  # [n, d]
        k = min(self.slots, len(outputs))
        
        # Weight by scores
        weights = mx.softmax(mx.array(scores[:k]))
        self.state = stacked[:k]
        
        # Broadcast: weighted mean
        broadcast = mx.sum(self.state * weights.reshape(-1, 1), axis=0)
        return broadcast


# =============================================================================
# TOKEN REGISTRY
# =============================================================================

class Registry:
    """Global vocabulary with token->cartridge mapping"""
    
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
            
            # Track co-occurrence for clustering
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
    """Orchestrates stem, cartridges, router, workspace"""
    
    def __init__(self):
        log.info("Initializing engine...")
        
        # Stem (frozen backbone)
        self.stem = Stem(vocab=32000, d=CFG.dim, depth=CFG.stem_depth, heads=CFG.heads)
        self.stem.set_dtype(mx.float16)
        mx.eval(self.stem.parameters())
        log.info("Stem initialized")
        
        # Components
        self.registry = Registry()
        self.carts: Dict[str, Cartridge] = {}
        self.router = Router(CFG.dim)
        self.workspace = Workspace(CFG.dim, CFG.workspace_slots)
        
        # Load state
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
    
    def spawn(self, tokens: List[int]) -> Cartridge:
        cart = Cartridge(self.stem, set(tokens))
        self.carts[cart.id] = cart
        self.registry.assign(tokens, cart.id)
        self.router.register(cart)
        log.info(f"Spawned {cart.id} with {len(tokens)} tokens")
        return cart
    
    def context_emb(self, tokens: List[int]) -> mx.array:
        if not tokens:
            return mx.zeros((CFG.dim,), dtype=mx.float16)
        x = mx.array([tokens[:CFG.seq_len]])
        h = self.stem(x)
        return mx.mean(h, axis=(0, 1))
    
    def train_step(self, batch: np.ndarray) -> Tuple[float, str]:
        x, y = batch[:, :-1], batch[:, 1:]
        ctx = self.context_emb(x[0].tolist())
        
        # Route
        routed = self.router.route(ctx, k=1)
        if not routed:
            unowned = self.registry.unowned()
            if unowned:
                cluster = self.registry.cluster(unowned[:50])
                cart = self.spawn(cluster)
            else:
                cart = self.spawn(list(range(2, min(100, len(self.registry.tokens)))))
        else:
            cart = self.carts.get(routed[0][0]) or list(self.carts.values())[0]
        
        loss = cart.train_step(x, y)
        self.router.update_sig(cart, ctx, loss)
        return loss, cart.id
    
    def maybe_spawn(self, tokens: np.ndarray, step: int):
        if step % CFG.spawn_interval != 0:
            return
        
        unowned = []
        for t in np.unique(tokens):
            if self.registry.tok2cart.get(int(t)) is None:
                unowned.append(int(t))
        
        if len(unowned) >= CFG.spawn_min_tokens:
            cluster = self.registry.cluster(unowned)
            if len(cluster) >= CFG.spawn_min_tokens:
                parent = list(self.carts.values())[0] if self.carts else None
                self.spawn(cluster)
    
    def generate(self, prompt: str, max_tokens: int = 30) -> str:
        tokens = self.registry.add(prompt)
        if not tokens or not self.carts:
            return "[No cartridges]"
        
        ctx = self.context_emb(tokens)
        routed = self.router.route(ctx, k=1)
        cart = self.carts.get(routed[0][0]) if routed else list(self.carts.values())[0]
        
        out_ids = cart.generate(tokens, max_new=max_tokens)
        words = [self.registry.tokens[t] if t < len(self.registry.tokens) else '<unk>' for t in out_ids]
        return ' '.join(words)
    
    def save(self):
        self.registry.save(os.path.join(CFG.library, 'registry.json'))
        for cart in self.carts.values():
            cart.save(CFG.library)


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
    
    def metrics(self) -> dict:
        with self.lock:
            return {
                'step': self.step, 'loss': self.loss, 'active_cartridge': self.active_cart,
                'tok_per_sec': self.tok_per_sec, 'num_cartridges': self.num_carts,
                'vocab_size': self.vocab_size, 'batch_size': self.batch_size,
                'memory_mb': self.memory_mb, 'memory_percent': self.memory_percent,
                'cpu_percent': self.cpu_percent, 'is_paused': self.paused,
                'training_mode': self.mode
            }

STATE = State()
ENGINE: Optional[Engine] = None


# =============================================================================
# DATA GENERATOR
# =============================================================================

def data_gen():
    import random
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models require training data.",
        "Neural networks approximate complex functions.",
        "Attention mechanisms improve model performance.",
        "Transformers revolutionized language processing.",
        "Embeddings map tokens to vectors.",
        "Backpropagation computes gradients efficiently.",
        "Apple Silicon has unified memory architecture.",
        "The brain uses modular specialized regions.",
        "Consciousness emerges from neural activity.",
    ]
    
    files = []
    if os.path.exists('training_data'):
        files = [os.path.join('training_data', f) for f in os.listdir('training_data') if f.endswith('.txt')]
    
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
            except:
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
    
    while True:
        try:
            if STATE.paused:
                time.sleep(0.05)
                continue
            
            # Fill buffer
            target = CFG.batch * (CFG.seq_len + 1) * 2
            while len(buf) < target:
                text = next(gen)
                toks = ENGINE.registry.add(text)
                buf.extend(toks)
            
            if len(buf) < CFG.batch * (CFG.seq_len + 1):
                time.sleep(0.01)
                continue
            
            # Build batch
            batch = []
            idx = 0
            for _ in range(CFG.batch):
                end = idx + CFG.seq_len + 1
                if end > len(buf):
                    break
                batch.append(buf[idx:end])
                idx += CFG.seq_len
            buf = buf[idx:]
            
            if len(batch) < CFG.batch:
                continue
            
            np_batch = np.array(batch, dtype=np.int32)
            
            # Train
            ENGINE.maybe_spawn(np_batch.flatten(), step)
            loss, cid = ENGINE.train_step(np_batch)
            
            if not np.isfinite(loss):
                continue
            
            step += 1
            tokens_done += CFG.batch * CFG.seq_len
            
            # Update metrics
            if step % 10 == 0:
                now = time.time()
                elapsed = now - t0
                tps = tokens_done / elapsed if elapsed > 0 else 0
                
                # System metrics
                mem_mb, mem_pct, cpu_pct = 0.0, 0.0, 0.0
                if HAS_PSUTIL:
                    proc = psutil.Process()
                    mem_mb = proc.memory_info().rss / (1024 * 1024)
                    mem_pct = proc.memory_percent()
                    cpu_pct = psutil.cpu_percent()
                
                with STATE.lock:
                    STATE.step = step
                    STATE.loss = float(loss)
                    STATE.active_cart = cid
                    STATE.num_carts = len(ENGINE.carts)
                    STATE.tok_per_sec = tps
                    STATE.vocab_size = len(ENGINE.registry.tokens)
                    STATE.batch_size = CFG.batch
                    STATE.memory_mb = mem_mb
                    STATE.memory_percent = mem_pct
                    STATE.cpu_percent = cpu_pct
                
                if step % 50 == 0:
                    tokens_done = 0
                    t0 = now
            
            # Save
            if step % CFG.save_interval == 0:
                ENGINE.save()
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            log.error(f"Training error: {e}")
            traceback.print_exc()
            time.sleep(1)


def chat_loop():
    global ENGINE
    while True:
        try:
            if not STATE.chat_q.empty():
                req = STATE.chat_q.get_nowait()
                STATE.paused = True
                time.sleep(0.05)
                
                try:
                    prompt = req.get('prompt', '')
                    conv = req.get('conversation', [])
                    
                    if conv:
                        parts = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in conv]
                        context = ' '.join(parts)
                    else:
                        context = prompt
                    
                    response = ENGINE.generate(context, max_tokens=req.get('max_tokens', 30))
                    STATE.resp_q.put({'text': response})
                except Exception as e:
                    log.error(f"Gen error: {e}")
                    STATE.resp_q.put({'error': str(e)})
                finally:
                    STATE.paused = False
            
            time.sleep(0.01)
        except Exception as e:
            log.error(f"Chat error: {e}")
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
    threading.Thread(target=training_loop, daemon=True).start()
    threading.Thread(target=chat_loop, daemon=True).start()
    log.info("Server started")


@app.get("/")
async def root():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"msg": "Infinite Cartridge System"}


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
    
    carts = [{"id": c.id, "tokens": len(c.tokens), "created": c.created, 
              "train_steps": c.steps, "strength": c.strength} for c in ENGINE.carts.values()]
    
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
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            
            while not STATE.resp_q.empty():
                STATE.resp_q.get()
            
            STATE.chat_q.put(data)
            
            t0 = time.time()
            while True:
                if not STATE.resp_q.empty():
                    await websocket.send_json(STATE.resp_q.get())
                    break
                if time.time() - t0 > 30:
                    await websocket.send_json({"error": "Timeout"})
                    break
                await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.error(f"WS error: {e}")


@app.get("/metrics")
def get_metrics():
    return STATE.metrics()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
