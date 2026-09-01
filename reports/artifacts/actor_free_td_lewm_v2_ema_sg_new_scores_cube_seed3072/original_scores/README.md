# Actor-Free TD-LeWM V2-EMA-SG Cube O50 archive

This directory is generated only from a validated six-training, eighteen-cell
formal EMA-SG bundle. It contains no checkpoints, dataset, video, or console
log. Every accepted epoch contains train and validation total loss, prediction
loss, online-reference MSE, online/EMA latent drift, base Hybrid TD and
method-specific Hybrid TD.

The LeWM one-step target is
`stop_gradient(EMA_world_model_next_latent)`; online histories, action
embeddings and predictions remain on the trainable online path.

Locked selection SHA-256: `e46ea81cce2e6a9a5df05ba04893b4181cbd8979340111a012c30f1efa2d7ee7`.
F-only/F+G use horizon 5; G-only uses horizon 1. Ranking is F+G only.
