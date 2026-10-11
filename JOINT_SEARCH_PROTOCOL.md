# IRASim mixed-precision search

## Reference

scmp_llm main: 59cb8f0f1164f638d931b472f3b64a8c5ab58ccb.

Initial allocation uses mp_per_row_chunk_calib8.py and the 80-iteration global
price solver from prc_offline_solve.py. Refinement uses mp_joint_refine.py (V11).
Integer projection and floor exchange are ported directly. Threshold movement
uses the per-row-chunk convention: a metric equal to the threshold stays in
the lower-length interval.

Kernel: ce3d7e5, including the scrambled-table cache identity fix.

## Search configuration

- Initial lengths: 8,10,12,14,16,20,24,28,32,40,48,56,64,80,96,112,128.
- Groups: row x 128 input channels.
- Metric: raw absmax, normalized over each complete linear call.
- Threshold sharing: operator x four layer buckets.
- Calibration sampling: 64 rows per call; 2048 stable-sort equal-count bins.
- Proxy cost: sampled MAC counts expanded to the full population.
- Initial allocation: monotone dynamic programming under a shared MAC-cycle price.
- Budget: measured mean variable-linear length 96, absolute tolerance 0.35.
- Initial budget projection: integer-length adjustment, up to three attempts.
- Ladder refinement: two rounds, floor increment 4, donor decrement 1.
- Threshold refinement: one round, mass step 0.02, 257-point rounded histograms.
- Threshold candidates: MLP-floor promotion and attention-linear-floor promotion.
- Acceptance: lower complete-task loss within the measured budget interval.
- Reference policies: uniform-96 and raw-error DP.
- Final selection: Fisher refinement incumbent compared against uniform-96.

The default recipe is shared by calibration, sensitivity measurement and video
confirmation. Timesteps come from the diffusion schedule; latent shape comes
from the configured model. Bridge sequence length is checked before evaluation.

Configurable fields include the initial ladder, integer target budget, length
bounds, layer buckets, search rounds, profile resolution and paired video seed.
The budget is independent of the initial ladder. Bounds apply to initial
allocation, supplied policies, budget projection and refinement candidates.
The uniform smoke test covers the selected budget and supports boundary values.
An explicit initial ladder is used exactly as supplied and must lie within the
bounds and span the budget. The default grid is clipped to the bounds, with an
endpoint added only when needed to span the budget; resolved lengths are recorded.

Refinement starts are selected by name: fis (default), raw, or parent with a
parent_policy JSON. RUN_JOINT_SEARCH.sh forwards additional command-line options.
Parent policies use ascending group_thresholds_v1 dispatch and numeric thresholds
for every active module. The schema accepts only lengths, dispatch, top-level
fractions and per-module group_thresholds/invert; validation covers calibration
loading and video confirmation.

Initial allocation determines occupied lengths. Refinement preserves the
parent's threshold topology and permits integer lengths from 1 through 128.
Budget projection accepts a candidate when its measured budget error decreases.
Refinement evaluates each changed active threshold candidate with the complete
model and updates occupancy from its actual forward pass.

## IRASim adapter

The frozen model predicts Gaussian noise on frames 2 through 16, with frame 1
as clean context. Complete-task loss is mean squared error against sampled
noise. Gaussian output VJPs estimate diagonal Fisher sensitivity using one
probe per case. SC forward values and FP straight-through derivatives provide
the sensitivity trajectory.

IRASim qkv/proj map to the attention-linear group; mlp_fc1/fc2 map to MLP.
The experiment's QK/AV, SQ scales, skip recipe and model weights remain fixed.
The reported budget denominator contains the variable MP-linear MACs.
Shape-based budget checks use the shared recipe's input chunk size. Runtime
budget recording has exclusive process-local ownership and restores the
original dispatch function when a stage completes or raises an exception.

Runtime settings: precision 8, bipolar halving, Owen bitrev, 64 scramble masks,
paired seed 3407. Fractional row dispatch remains available for historical
table reproduction. The current search uses numeric per-group thresholds.

An optional expand_at policy supplies the sensitivity/activation trajectory
for a new calibration pass.

## Data and confirmation

Calibration: two existing episodes, three stratified timesteps per episode.
Noise confirmation: separate episodes, six stratified timesteps per episode.
Video confirmation: PNDM-50, paired seed 3407.

Calibration selects and freezes the final policy. Noise and video confirmation
report the frozen policy alongside uniform. Reports include task loss,
PSNR/SSIM/latent L2 and measured budget status as applicable.
search_policy.json remains the frozen calibration policy. confirmation_status.json
separates evaluation completion from budget eligibility: passed requires noise
confirmation and both video arms to meet their budget checks. A new confirmation
attempt starts with pending status. Superseded selected.json artifacts move into
the run's superseded directory, retaining a recoverable copy.

The original n=10 episodes form the historical reproduction set. Episode-level
split checks separate it from calibration and confirmation. The present
calibration size is a pilot; independent confirmation measures generalization.

## Artifacts and stages

- smoke: native-loss equivalence, uniform dispatch identity, STE forward
  identity and finite sensitivity.
- search: initial allocation, budget projection, ladder/threshold refinement;
  save the completed state and frozen search_policy.json.
- noise-validate: independent noise confirmation.
- video-validate: paired video confirmation of search_policy.json.

RUN_JOINT_SEARCH.sh selects one stage per invocation. The n=10 reproduction
run is a separate action requiring approval.

Source, kernel, checkpoint metadata, split files, SQ scales, latent tensors,
action tensors and sampled noise are recorded in provenance. A matching GPU
smoke is required for calibration. CPU tests cover solver/controller logic;
GPU smoke establishes runtime compatibility.
The mixed smoke checks three occupied lengths (32, 79, 128), equal-to-cut
dispatch, bias and SmoothQuant, and compares fused results with independent
chunk products. Calibration requires this check in addition to uniform identity.

Checkpoint contents, runtime sources and configuration files are hash-checked;
runtime software versions are recorded and checked across stages.
GPU model, compute capability, device memory, CUDA/cuDNN and NVIDIA driver
versions are also recorded and checked. Hardware identity participates in
video cache fingerprints; unavailable driver metadata is recorded as null.
Source hashes include video generation and dataset/action conversion code.
Numerical settings are pinned before kernel imports: the hardware mask cap is
64, device-dependent upstream kernel dispatch is used, and FP32 matmul and cuDNN
TF32 are disabled in both calibration and paired video evaluation. Effective
settings are recorded; inherited policy, kernel-ablation and trace overrides
are cleared before each stage initializes.
Video run fingerprints also cover original pixels, annotations, decoder and
scheduler files, the frozen policy and the recorded video seed. Reuse requires
a completed run and matching result-file hashes. Incomplete or changed attempts
receive a fresh directory; prior artifacts are preserved.

Hypothesis: the V11 allocation and refinement improves noise-prediction loss
at equal measured linear cost. Calibration loss determines the search winner;
independent noise and video results assess transfer to held-out cases.

Status: local implementation; GPU validation pending.

Case-level parallel execution uses a persistent spawn-process pool (--workers).
Each worker owns its model and hook, and evaluates one frozen case per task.
Results retain serial case order; loss, occupancy and profiles use original
case weights. Curve measurements use the global case-weight denominator.
GPU smoke checks parallel predictions and budgets against serial evaluation.
