# Group recalibration

The earlier group pilot reused per-row calibration. Keep those outputs only
as a migration diagnostic; they are not an optimized group result.

The new run fixes the kernel at ce3d7e5, chunk width 128, precision 8,
bipolar halving, existing SmoothQuant scales and skip recipe, PNDM-50,
generation seed 3407. Calibration episodes 0 and 1000, holdout episodes 1001
and 1005, and the original ten test episodes are disjoint. The existing SQ
recipe is held fixed rather than claiming newly independent SQ calibration.

First search: each enabled linear chooses one or two lengths from
{32,64,96,112,128}, fractions from {0.25,0.5,0.75}, and rank direction.
The grid is a bounded candidate space, not a claimed globally optimal ladder.
Uniform-96 is included. Every policy is measured with the complete deployed
chunked matmul, including SQ and cross-chunk error cancellation. Group ranking
remains raw absmax, exactly as at runtime; oracle error assignments are not
converted into fractions. Capture eight rows at steps 0,25,49 of each FP
teacher trajectory. This is a small screening calibration, not full coverage.

Minimize each module's calibration relative output MSE subject to realized
average length <=96, checking rounding for sampled and original row counts.
The per-module constraint is stronger than a global MAC-weighted constraint;
cross-module budget transfer is outside this first search. The heldout score
is relative output MSE weighted by deployment output elements. If the complete
proposed policy does not beat uniform on this independent activation holdout,
select uniform for every module. No test metrics select the policy.

Then evaluate selected and uniform on exactly the original ten trajectories,
same sampler, seed and preprocessing. Primary image metric: paired mean PSNR.
Also report SSIM, latent L2 and per-trajectory wins. A gain needs positive PSNR
without degraded mean SSIM or latent L2; otherwise report no supported gain.
Reconstruction-error fallback guarantees only its measured local objective,
not image PSNR or performance on unseen data. No test-set oracle fallback.

Keep the historical Table VI reproduction unchanged: its 7.58-bit result is
the timestep schedule, not this static group allocation.
