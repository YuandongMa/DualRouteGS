# Sources and third-party notices

The public-facing package, model classes, configuration and execution entry points
are named DualRouteGS. Source provenance is retained here and in original source
headers. This is a new research implementation, not an official release from any
upstream author and not a claim of sole authorship over third-party code.

## Radiative rendering backend

- Source: [Ruyi-Zha/r2_gaussian](https://github.com/Ruyi-Zha/r2_gaussian)
- Inspected and copied revision: `f2579bfddd9aac009cb797c8503bef8119bbd022`.
- Copied subtree: `r2_gaussian/submodules/xray-gaussian-rasterization-voxelization`.
- Bundled location: `third_party/ray_gaussian`.
- Original license: `third_party/licenses/radiative_backend_LICENSE.md`, also
  reproduced in the backend directory. Original Inria / GRAPHDECO headers remain.
- The upstream license specifies noncommercial research and evaluation use.
  This bundle is not represented as wholly MIT-licensed.
- GLM headers are retained at pinned revision
  `33b4a621a697a305bc3a7610d290677b96beb181` from
  [g-truc/glm](https://github.com/g-truc/glm); its `copying.txt` is included.

Modifications for this project:

1. `cuda_rasterizer/forward.cu` and `backward.cu`: remove the alpha `< 1e-5`
   early exit so zero-forward-density candidates retain existence-gate gradients.
2. `cuda_voxelizer/forward.cu` and `backward.cu`: remove alpha-based truncation
   and use a 3-sigma Mahalanobis support (`power >= -4.5`). The forward voxelizer
   also derives a conservative radius from covariance when explicit scales are
   absent, avoiding an invalid scales pointer in the precomputed-covariance path.
3. Backend `__init__.py`: add `DUALROUTEGS_GATE_SAFE` compatibility marker.
4. Retain only build-relevant GLM files; upstream documentation/test folders are
   omitted. Original source headers and license files are not renamed away.

The supplied high-level renderer adapter, attenuation Gaussian container,
reference ray integrator and model pipeline are written for this project.
Legacy `opacities` names are confined to the external ABI and carry attenuation
density here; RGB alpha compositing is not used.

## Gaussian diffusion architectural reference

- Source: [caiyuanhao1998/Open-DiffusionGS](https://github.com/caiyuanhao1998/Open-DiffusionGS)
- Inspected revision: `5b57ca2aa5cedf1fd3c4f4d9b5970d7f869bf8d4`.
- Inspected modules: `models/denoiser/denoiser_scene.py`,
  `models/scheduler/ddim_scheduler.py`, `systems/diffusion_gs_system_scene.py`,
  and the Gaussian renderer interface under `diffusionGS/`.
- The upstream distribution states it is a reimplementation differing from the
  original Adobe-developed version.
- Architectural ideas retained: camera/ray-conditioned image tokens, Gaussian
  prediction inside a denoiser, and rendered clean-image supervision.
- No upstream model checkpoint or RGB model module is copied into the primary
  package. The scheduler and attenuation denoiser are newly implemented from the
  manuscript equations. No weight compatibility is claimed.
- The upstream MIT license is included at
  `third_party/licenses/diffusion_reference_MIT.txt` for provenance, rather than
  being applied to the radiative backend.

## Scientific references

Ruyi Zha, Tao Jun Lin, Yuanhao Cai, Jiwen Cao, Yanhao Zhang, Hongdong Li.
*R²-Gaussian: Rectifying Radiative Gaussian Splatting for Tomographic
Reconstruction.* NeurIPS, 2024.

Yuanhao Cai et al. *Baking Gaussian Splatting into Diffusion Denoiser for Fast
and Scalable Single-stage Image-to-3D Generation and Reconstruction.* ICCV, 2025.

Yuandong Ma and Yong Zhang. *DualRouteGS: Coordinated State and Gaussian
Routing for Single-View Transmissive Reconstruction.* User-provided manuscript,
`DualRouteGS (8).pdf`. No venue/publication status is assigned here.
