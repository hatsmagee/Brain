#!/bin/bash
# Infinite Cartridge System - Brain-Inspired Neural Architecture
# Optimized for Apple Silicon (M1/M2/M3)
#
# Usage:
#   chmod +x run.sh && ./run.sh
#
# Prerequisites (install once):
#   pip install mlx numpy fastapi uvicorn[standard] sse-starlette psutil

set -e

# Create training data directory if needed
mkdir -p training_data

# Check for training files
if [ -z "$(ls -A training_data 2>/dev/null)" ]; then
    echo "📁 No training data found. Creating sample..."
    cat > training_data/sample.txt << 'EOF'
The brain is organized into specialized modules with dense internal connections.
Neural networks learn by adjusting connection weights through backpropagation.
Attention mechanisms help models focus on relevant parts of the input.
The prefrontal cortex acts as a connector hub coordinating specialized regions.
Memory consolidation occurs during sleep through synaptic replay.
Plasticity allows the brain to reorganize and form new connections.
Visual processing begins in V1 and flows through increasingly abstract regions.
Language emerges from distributed networks spanning multiple cortical areas.
The hippocampus is crucial for forming new episodic memories.
Emotions are processed by the amygdala and influence decision making.
Motor control involves the cerebellum for fine coordination.
The thalamus acts as a relay station filtering sensory information.
Consciousness may emerge from global workspace broadcasting.
Neurons that fire together wire together through Hebbian learning.
Sleep deprivation impairs cognitive function and emotional regulation.
The default mode network is active during rest and introspection.
Mirror neurons may play a role in understanding others actions.
Neurogenesis continues in certain brain regions throughout life.
Dopamine pathways are involved in reward and motivation.
The autonomic nervous system regulates involuntary functions.
EOF
    echo "✅ Sample training data created"
fi

echo "🧠 Starting Infinite Cartridge System..."
echo "   Dashboard: http://localhost:8000"
echo "   Press Ctrl+C to stop"
echo ""

python3 cartridge_system.py
