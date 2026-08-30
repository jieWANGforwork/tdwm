# Actor-Free TD-LeWM V0 formal Cube O50 summary

`formal_o50_summary.json` is a compact, immutable index of the completed V0
`C/D/F/G1/G2/G3 × F-only/G-only/F+G` evaluation matrix.  The values were
recomputed from each raw `results.json` file under the recorded AutoDL source
root; every entry includes the SHA-256 of that raw result file.

The 18 evaluations use training seed 3072, planning seed 42, the same 50
start-goal pairs, goal offset 50, and selection SHA-256
`e46ea81cce2e6a9a5df05ba04893b4181cbd8979340111a012c30f1efa2d7ee7`.

This compact index does not replace the server-side raw outputs.  It exists so
the V0 values used in the V0/V1 comparison are explicit, reviewable, and tied
to their source-file hashes.
