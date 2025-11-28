# 🧠 Infinite Cartridge System

Brain-inspired modular neural network for Apple Silicon. Millions of tiny specialized "cartridges" (~100KB each) share frozen universal "stem" layers and differentiate via lightweight adapters.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      INPUT QUERY                            │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  STEM (Frozen Backbone)                     │
│   Universal layers trained once, shared by ALL cartridges   │
│   Like V1/A1 in cortex - early sensory processing           │
│   ~5MB, cached in SLC                                       │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                 ROUTER (Prefrontal Cortex)                  │
│   Selects 2-4 relevant cartridges via learned signatures    │
│   Hebbian learning: successful paths strengthen             │
└──────┬──────────────┬──────────────┬──────────────┬─────────┘
       │              │              │              │
       ▼              ▼              ▼              ▼
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ Cart A   │   │ Cart B   │   │ Cart C   │   │ Cart D   │
│ ~100KB   │   │ ~100KB   │   │ ~100KB   │   │ ~100KB   │
│ LoRA +   │   │ LoRA +   │   │ LoRA +   │   │ LoRA +   │
│ Signal   │   │ Signal   │   │ Signal   │   │ Signal   │
└────┬─────┘   └────┬─────┘   └────┬─────┘   └────┬─────┘
     │              │              │              │
     └──────────────┴──────┬───────┴──────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              GLOBAL WORKSPACE (Consciousness)               │
│   Top outputs compete for slots, winners broadcast to all   │
│   Like thalamo-cortical binding                             │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
                   OUTPUT
```

## Key Concepts

| Component | Brain Analog | Function |
|-----------|--------------|----------|
| Stem | V1, A1, early cortex | Universal feature extraction (frozen) |
| Cartridge | Provincial hub | Specialized domain expert |
| LoRA Adapter | Epigenetic marks | Lightweight specialization |
| Diff Signal | BMP/Wnt signaling | Domain identity vector |
| Router | Prefrontal cortex | Expert selection |
| Workspace | Global workspace | Conscious binding |
| Hebbian Update | Synaptic plasticity | "Fire together, wire together" |

## Memory Footprint

- **Stem (shared)**: ~5MB (loaded once)
- **Per cartridge**: ~100KB
- **Router + workspace**: ~500KB
- **Active set (4 carts)**: ~400KB

**1M cartridges = 100GB storage, but only ~6MB active at once**

Fits comfortably in M1 Max's 48MB SLC for streaming inference.

## Quick Start

```bash
# Install dependencies (macOS with Apple Silicon)
pip install mlx numpy fastapi uvicorn[standard] sse-starlette psutil

# Run
chmod +x run.sh
./run.sh

# Open dashboard
open http://localhost:8000
```

## Dashboard Features

- **Training Metrics**: Step, loss, active cartridge
- **Performance**: Tokens/sec, batch size, vocab size
- **System**: Memory, CPU usage
- **Chat**: Interactive generation
- **Visualization**: Radial cartridge graph
- **Registry**: All spawned cartridges

## Training Data

Put `.txt` files in `training_data/` directory. The system will:
1. Tokenize text
2. Build vocabulary
3. Spawn cartridges for token clusters
4. Train cartridges on their owned tokens

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard UI |
| `/stream/metrics` | GET (SSE) | Real-time metrics |
| `/cartridges` | GET | List all cartridges |
| `/training/pause` | POST | Pause training |
| `/training/resume` | POST | Resume training |
| `/training/mode/{cpu\|gpu}` | POST | Switch device |
| `/ws/chat` | WebSocket | Chat interface |

## Cartridge Lifecycle

1. **Spawn**: New domain detected → cluster tokens → create cartridge
2. **Train**: Receive batches → update LoRA + head (stem frozen)
3. **Route**: Query arrives → router scores signatures → top-k selected
4. **Bind**: Outputs compete in workspace → winners broadcast
5. **Hebbian**: Low loss → strengthen connection; high loss → weaken
6. **Save**: Periodic checkpoint to `cartridge_library/`

## Configuration

Edit `Config` class in `cartridge_system.py`:

```python
@dataclass 
class Config:
    dim: int = 256              # Model dimension
    stem_depth: int = 4         # Frozen backbone layers  
    adapter_rank: int = 16      # LoRA rank per cartridge
    cart_vocab: int = 1024      # Tokens per cartridge
    active_k: int = 4           # Active cartridges per query
    batch: int = 32
    seq_len: int = 64
    lr: float = 3e-4
```

## Scaling to Millions

```
cartridge_library/
├── registry.json           # Token mappings
├── domain_000/             # Science
│   ├── cart_0000/
│   │   ├── weights.safetensors  (100KB)
│   │   └── meta.json
│   ├── cart_0001/
│   └── ...
├── domain_001/             # History
└── ...

16 domains × 256 topics × 256 experts = 1M cartridges
```

Only active cartridges loaded; rest streamed on demand.

## License

MIT
