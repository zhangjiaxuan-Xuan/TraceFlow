# Frozen protocols

## LIBERO

`run_libero_best_98_3.sh` deliberately runs four independent configurations. Spatial and Goal use the complete B6500 positive demonstration bank and k+=8. Object uses the deterministic B50-per-task bank with the N497+C-pi116 failure bank at k+=16/k-=8. LIBERO-10 uses the B50/k16 raw-best selection with the same negative contract.

All four use Pi0.5-LIBERO, V1 full-interval additive capped guidance, ten flow steps, seed 7, 50 episodes per task, no LCM, and no prior interpolation. The arithmetic envelope is 98.3%; it is not a unified run.

For comparison, the canonical unified V1 configuration uses B50/k16 plus N+C-pi/k8 for all four suites and produced 491/500, 500/500, 489/500, and 480/500 respectively: 1960/2000 (98.0%).

LIBERO-Plus-10 evaluates all 2,519 registered variants once with seed 7 and the LIBERO-10-only positive and negative slices. No Plus trajectory is included in memory. Its registered result is 2043/2519 (81.10%).

## RoboMemArena

| Entrypoint | Tasks | Bank | Head | Guidance |
|---|---|---|---|---|
| Sequence | 1,2,3,22 | Extra8 joint | Upper | V0 |
| Transferring | 18,19,25,26 | Extra8 | Fusion | V1 |
| Counting | 6,7,8,9,10,15,16 | suite-only Full26 slice | Fusion | V1.1 suite gate |
| Occlusion | 4,5,11,12,13,14,17,20,21,23,24 | suite-only Full26 slice | Fusion | V1.1 suite gate |

Production defaults are two GPUs, Upper/Lower batch 32, 64 environment workers, and the frozen episode/seed counts in `configs/release`. Debug overrides change throughput and sample size, not the algorithmic config.

## TraceBankStack

Transferring starts from the immutable Extra8 bank; LIBERO-10 begins with its recorded collector. Thereafter, round artifacts are append-only by branch. Success admits successful rollouts, failure admits unsuccessful rollouts, and joint admits both while keeping the physical positive and negative stores separate. Resuming requires the recorded config and all preceding round outputs to match.

A round with no newly admitted success retains the inherited positive bank. A
round with no newly admitted failure runs without a negative bank. Both cases
are recorded explicitly and remain resumable; an empty outcome class is not an
evaluation failure.
