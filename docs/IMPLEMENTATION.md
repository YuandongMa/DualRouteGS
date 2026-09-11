# Equation-to-Code Mapping and Implementation Scope

This project implements the method and experimental settings described in the supplied `DualRouteGS (8).pdf`. It is a trainable research implementation, not a claim to the original experimental repository. Evaluation code does not contain pre-filled scores from the manuscript tables.

## Equation-to-Code Mapping

| Manuscript component | Equation | Implementation |
| --- | --- | --- |
| Log transmission | (1) | `scripts/calibrate_projection.py`, `geometry.Camera` |
| Target noising, noise recovery from rendered x0, and DDIM updates | (2)-(4) | `DiffusionSchedule`, `reconstruct` |
| Centers, rotations, anisotropic scales, and nonnegative attenuation | (5) | `GaussianSet`, with `Sigma=R diag(s^2) R^T` |
| Target-first and condition-second tokens; geometry conditioning, RoPE, and AdaLN | (6)-(7) | `DualRouteGS`, `DualRouteBlock` |
| Log-SNR reliability with a positive slope | (8) | `PhaseReliability`; softplus enforces a positive slope |
| DWConv, negative A, and exact ZOH coefficients | (9)-(10) | `PhaseAwareStateRouting.forward`, using `expm1` for numerical stability |
| Stop-gradient dominant-pattern memory across network depth | (11)-(14) | Reset on each forward call and passed between layers; no caching across samples or diffusion steps |
| Joint confidence from amplitude, innovation, and phase | (15)-(16) | `confidence_coeff`, `phase_bias` |
| Base subspaces, phase-dependent amplitudes and thresholds, and ST masks | (17)-(24) | PSR and `models/scan.py` |
| Purified memory and sparse foreground queries | (25)-(27) | Dense ST relaxation during training; gather/attention/scatter at inference; direct bypass for empty query sets |
| Candidate bank, shared allocation score, and strictly ordered thresholds | (28)-(31) | `InnovationGuidedAllocator`, with cumulative softplus candidate gaps |
| Minimum capacity, ST existence variables, and hard selection | (32)-(35) | `Allocation.relaxed()`, `Allocation.active()` |
| Reconstruction at two supervised views, SSIM, alignment, and budget losses | (36)-(40) | `reconstruction_loss`, `train.py` |
| Three-dimensional attenuation field and three-sigma truncation | (41) | Analytic reference voxelization and the modified CUDA voxelizer |
| Actual number of active Gaussians | (42) | Inference traces and evaluation files, computed from actual gate outputs |

## Explicitly Retained Experimental Settings

The full configuration uses eight layers, width 512, and eight retrieval heads; `R=8` and `r_min=2`; minimum write amplitudes of 0.05 in early phases and 0.15 in late phases; and a memory decay factor of 0.9. A 512 x 512 anchor grid with `K_max=4` produces 1,048,576 candidates before routing. Training uses 80K optimizer updates, learning rate 2e-5, weight decay 0.05, 5K warmup updates, cosine decay, batch size one, gradient accumulation over four steps, and CUDA bfloat16 mixed precision. Inference defaults to 20 DDIM steps. The loss weights are 1/1/0.2/0.1/0.02, with an upper additional-candidate ratio of 0.6.

## Choices for Details Not Fully Specified in the Manuscript

| Detail | Current choice | Implication for reproduction |
| --- | --- | --- |
| Patch size | 16 | Affects sequence length and attention cost |
| Total state width and B tensor | Width 512; channel-wise diagonal B with a learned input projection | One dimensionally consistent realization of the recurrence; equivalence to the original code is not assumed |
| Position encoding | Separate 2D input RoPE for target and condition streams | Rotation dimensions and frequency details are not specified in the manuscript |
| Diffusion schedule | 1,000-step cosine schedule | The manuscript specifies a predefined schedule without explicit beta values |
| Phase-dependent threshold offsets | 0.15 for both rank and query thresholds | Can be tuned on validation data |
| Gate smoothing | Rank/query sharpness of 12; existence temperature of 0.5 | Affects ST gradients and routing density |
| Lightweight Gaussian decoder | 64 channels, bilinear upsampling, and depthwise convolution | The original decoder's layer and channel counts are not specified |
| Gaussian center prior | Uniform candidates along the target ray's intersection with the scene box, plus bounded learnable center offsets | Provides trainable candidates at distinct initial spatial locations; initialization rules are not specified in the manuscript |
| Scale and attenuation initialization | Configured bounded positive scales and softplus kappa | Affects training stability and should reflect calibration and data scale |
| Retrieval during training | Dense query computation, hard bypass in the forward pass, and ST relaxation in the backward pass | Inference uses actual sparse gathering; equivalent training FLOP savings are not claimed |
| Scan implementation | Associative prefix scan with O(N log N) work | No custom fused state kernel is provided; its performance is not equivalent to a claimed linear-time state kernel |
| Sparse state execution | Compact active tokens within each subspace, preserve their original order, then restore the state sequence | Mathematically equivalent to the hard-mask recurrence; sorting tokens by rank must not change recurrence order |
| Supervision | Denoising and held-out terms plus SSIM, following (37) | No additional independent input-view loss or volume supervision is introduced |
| Data split manifests | Externally supplied explicit patient/object splits | Original subject IDs were not provided, so manuscript split identities are not fabricated |

Instantiating the full configuration yields **46,551,209 parameters**, compared with 112.1M reported in the manuscript table. The difference results from implementation choices for undisclosed details; changing displayed counts would not resolve it. This project does not claim to reproduce the manuscript's latency, FLOPs, memory use, quality, or duration of an 80K-update training run.

## Renderer Relationship

The reference renderer analytically integrates each Gaussian along a finite ray `o+s*d, 0<=s<=L`. With `Q=Sigma^-1`, `u=o-c`, `a=d^T Q d`, `b=d^T Q u`, and `c0=u^T Q u`, the contribution is:

`kappa * exp[-(c0-b^2/a)/2] * sqrt(2*pi/a) * (erf((a*L+b)/sqrt(2*a))-erf(b/sqrt(2*a)))/2`.

It uses attenuation integration without RGB, spherical harmonics, or alpha compositing. It retains existence-gate gradients for all training candidates and is intended for numerical checks and small experiments, rather than high-performance splatting.

The CUDA backend uses radiatively corrected affine ray-space projection and tile splatting. It is not strictly equivalent to analytic finite-ray integration for every geometry and Gaussian scale. The main adapter supports centered cone-beam geometry; other geometries produce an explicit error to avoid silently applying an incompatible projection matrix.

To preserve existence-gate gradients for inactive candidates, early exits based on very small alpha values are removed from both the CUDA forward and backward passes, while geometry and tile culling remain. Voxelization uses three-sigma ellipsoidal support, `power >= -4.5`, instead of attenuation-dependent truncation. Three CUDA tests are included. They were skipped because no GPU was available and must be run after compilation on the target GPU.

## Evaluation Scope

SSIM uses an explicit Gaussian-window implementation, and 3D SSIM uses three-dimensional Gaussian filtering. Boundary handling differs between libraries, so compared methods must use the same implementation. Optional NSD uses equally weighted boundary voxels, not surfel-area weighting. clDice depends on the selected version of three-dimensional skeletonization. Without evaluation masks, the code does not invent nodule Dice, NSD, or material-identification results.

Inference timing includes DDIM, decoding, hard routing, and canonical target rendering. It excludes data loading, subsequent rendering at each evaluation target, and volume evaluation. The runtime of a single network forward pass is not reported as the complete reconstruction time.
