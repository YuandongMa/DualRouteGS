# DualRouteGS

**Coordinated State and Gaussian Routing for Single-View Transmissive Reconstruction**

A trainable research implementation based on the supplied DualRouteGS manuscript. The network, modules, configurations, and execution interfaces use the DualRouteGS name and are organized around **PSR + IGA**. The input is **one calibrated scalar X-ray transmission projection and its acquisition geometry**. The output is a three-dimensional Gaussian attenuation field in global coordinates, which supports novel-view projection rendering and volume reconstruction.

This codebase was developed from the manuscript and public implementations. **It is not the authors' original experimental repository and does not include manuscript-trained weights or reproduced benchmark results.** Equations, gradients, and the complete data workflow have been checked on CPU. CUDA source code and adapters are included, but they have not been compiled or tested in the current environment because no GPU was available. See the [verification report](verification/REPORT.md) for the actual validation record.

## 1. Components and Entry Points

| Component | Entry point |
| --- | --- |
| DualRoute network, geometry encoding, AdaLN, and 2D RoPE | `dualroutegs/models/dualroute.py` |
| Phase reliability, memory-orthogonal innovation, and amplitude/subspace/query selection | `dualroutegs/models/psr.py` |
| Order-preserving sparse scanning and associative scanning for training | `dualroutegs/models/scan.py` |
| Ordered candidate thresholds, minimum capacity, and differentiable existence gates | `dualroutegs/models/iga.py` |
| Attenuation Gaussian parameters and covariance | `dualroutegs/gaussians.py` |
| Transmission rendering: CUDA backend and analytic PyTorch reference | `dualroutegs/rendering/` |
| Forward diffusion and DDIM driven by rendered predictions | `dualroutegs/diffusion.py` |
| Data loading and patient/scene split checks | `dualroutegs/data/dataset.py` |
| Training, inference, and evaluation | `train.py`, `infer.py`, `evaluate.py` |

See [IMPLEMENTATION.md](docs/IMPLEMENTATION.md) for the equation-to-code mapping and implementation assumptions, and [DATA.md](docs/DATA.md) for the data format.

## 2. Installation

Python 3.10 or later is required. Validation used Python 3.12 and PyTorch 2.5.1 CPU. Use a dedicated virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CPU validation environment; use a compatible CUDA build of PyTorch for GPU training.
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
```

For an NVIDIA GPU, install a CUDA-enabled PyTorch build and a compatible CUDA toolkit with nvcc, then compile the bundled backend:

```bash
python -m pip install ninja
python scripts/install_backend.py
python -m unittest discover -s tests -v
```

The CUDA installation command uses `--no-build-isolation` to access PyTorch from the current environment. No additional model weights are required. `third_party/ray_gaussian` includes the necessary C++/CUDA source files and GLM headers.

## 3. Minimal End-to-End Example

Run the following commands from the project root:

```bash
python -m unittest discover -s tests -v
python scripts/make_demo.py --output data/demo
python train.py --config configs/smoke.yaml --device cpu --output runs/smoke
python infer.py --checkpoint runs/smoke/last.pt \
  --input data/demo/example_input.npy \
  --camera data/demo/example_camera.json \
  --device cpu --volume-resolution 16 --output outputs/demo
```

This workflow performs only three optimizer updates to check the execution pipeline. The synthetic examples and resulting reconstructions **must not be used as manuscript experiments or evidence of reconstruction quality**.

Exported files:

| File | Contents |
| --- | --- |
| `gaussians.npz` | Hard-selected `means / scales / rotations / kappa` |
| `canonical_projection.npy` | Projection at the canonical virtual target view, restored to the input projection scale |
| `condition_projection.npy` | Reprojection under the input camera geometry |
| `volume.npy` | Attenuation field at the requested resolution, with xyz array axes |
| `metadata.json` | Coordinate scale, random seed, sampling steps, and measured routing statistics |

## 4. Training with Your Data

Prepare a manifest following [DATA.md](docs/DATA.md). Existing scanner scene directories can be converted with:

```bash
python scripts/convert_scanner.py --catalog data/catalog.json --output data/calibrated
python train.py --config configs/paper.yaml \
  --manifest data/calibrated/manifest.json --device cuda --output runs/full
```

`paper.yaml` retains the manuscript's explicitly stated settings: eight layers, width 512, eight retrieval heads, eight state subspaces, two base subspaces, a 512 x 512 anchor grid, up to four Gaussians per anchor, 80K optimizer updates, four-step gradient accumulation, and the stated loss weights. Details omitted from the manuscript are specified as explicit configuration choices. The configuration name does not imply that the reported parameter count or performance has been reproduced exactly.

Use `development.yaml` to debug at smaller image resolutions and anchor counts with the reference renderer. Its computational cost scales with the number of rays multiplied by the number of Gaussians; **use the CUDA backend for the 512 x 512 anchor configuration**. The CUDA adapter currently supports centered cone-beam cameras. The reference backend also supports checks with parallel beams and offset detectors.

Resume training with:

```bash
python train.py --resume runs/full/last.pt --device cuda --output runs/full
```

Training logs are stored in `train.jsonl`. Checkpoints include the network, optimizer, actual configuration, iteration count, and major RNG states. The data iterator is recreated when training resumes, so sample-by-sample bitwise equivalence to uninterrupted training is not guaranteed. Use the separate validation command below for model selection; the training script does not automatically inspect the test set.

## 5. Single-View Inference and Novel-View Rendering

```bash
python infer.py --checkpoint runs/full/last.pt \
  --input data/input_projection.npy --camera data/input_camera.json \
  --device cuda --volume-resolution 256 --output outputs/case001
python scripts/render_views.py --gaussians outputs/case001/gaussians.npz \
  --geometry data/novel_geometry.json --size 256 --backend cuda \
  --device cuda --output outputs/case001/novel_views.npz
```

`infer.py` accepts exactly one measured projection. The canonical target canvas is initialized from noise; the interface does not accept additional measured target images.

## 6. Validation, Testing, and Ablations

```bash
python -m pip install -e '.[metrics]'
python evaluate.py --checkpoint runs/full/last.pt --split val --volume --device cuda
# Run test evaluation only after fixing the model and hyperparameters.
python evaluate.py --checkpoint runs/full/last.pt --split test --volume --device cuda
```

Projection metrics include PSNR, SSIM, and RMSE. `--lpips` enables optional LPIPS evaluation, whose dependency loads the required evaluation-network weights. Volume metrics include PSNR, 3D SSIM, and RMSE. Results are aggregated over target views, scenes, and independent patients/objects, with group-level bootstrap confidence intervals. `scripts/paired_statistics.py` provides paired Wilcoxon tests with Holm correction. `scripts/evaluate_regions.py` provides region PSNR/MAE, Dice, boundary-voxel NSD, and clDice. Segmentation masks come from an external fixed evaluation workflow and are not reconstruction inputs.

`configs/ablations/` includes static attention, static state routing, PSR only, static state routing with IGA, and configurations that disable phase, amplitude, subspace, or query routing. Train each configuration separately using `train.py --config ...`, then evaluate all variants on identical data splits.

## 7. Source Attribution

Dependency sources, pinned revisions, modification scopes, and original licenses are documented in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `third_party/licenses/`. The main project is organized independently, while the dependencies' copyright notices and required attribution are retained.
