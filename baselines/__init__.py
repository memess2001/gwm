"""Control-group world models for every kill test (HANDOFF section 4): plain MLP / conv world models in three
variants with identical encoder and transition architectures, on a 4-size parameter ladder.

    baselines.data      windowed datasets from zoo Trajectories (obs / actions only), train-split normalisation
    baselines.mlp_wm    WorldModel (E, F, D), variants "recon", "jepa-sym", "jepa-sg" ("jepa" = jepa-sym), size
                        ladder S / M / L / XL, FLOPs
    baselines.train     training CLI:  python -m baselines.train --world pendulum --obs angle_cs --variant recon ...
    baselines.encode    load a checkpoint, latents z_t for a Trajectories object (for the pre-registered analysis)
    baselines.ledger    append rows to results/ledger.md

Nothing is imported here, so ``baselines.ledger`` works without torch.
"""
