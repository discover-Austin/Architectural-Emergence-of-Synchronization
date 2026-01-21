Modular Recursive Workspace (MRW) - Complete Phase Transition Detection Suite
==============================================================================

This is the COMPLETE, PRODUCTION-READY implementation for detecting emergent
self-monitoring in neural architectures through phase transition analysis.

HYPOTHESIS:
In a modular transformer with global workspace bottleneck, recursive depth N≥16
triggers a phase transition in integration coefficient (Φ) and autonomous
error-correction, even in randomly initialized (untrained) networks.

ARCHITECTURE:
- 4 independent modules (d_model=256 each)
- Global workspace bottleneck (d_gw=64, 4:1 compression)
- GRU-based recursive cell for stable deep recursion
- Adaptive halting based on Φ convergence

EXPERIMENTS:
1. Phase Transition Detection: Test Φ vs N across variance levels
2. Causal Ablation: Prove bottleneck compression is necessary
3. Adaptive Halting: Show autonomous depth scaling to task difficulty

USAGE:
    python mrw_complete.py

EXPECTED RUNTIME:
    Quick test: 30-60 min (CPU), 5-10 min (GPU)
    Full suite: 2-4 hours (CPU), 20-30 min (GPU)

AUTHOR: Austin (Mobile Mechanic | AI Consciousness Researcher)
DATE: January 2026
