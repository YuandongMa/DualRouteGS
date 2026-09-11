# Data and Coordinate Conventions

## Model Inputs

At inference, each sample supplies **one nonnegative scalar log-transmission image of shape [H,W]** and its calibrated acquisition geometry. Other views of the same object may provide training supervision, but they are not additional measured conditions.

RGB pseudocolors generally do not represent attenuation coefficients that can be integrated directly. If only colorized security-screening images are available, obtain the original scalar channel or an invertible color calibration from the vendor. Direct grayscale conversion changes the physical meaning and is not performed automatically. Multi-energy or material channels require a separately defined forward physical model.

Convert raw detector intensities using `P=-log((raw-dark)/(flat-dark))`. The script `scripts/calibrate_projection.py` implements this conversion and reports the fraction of clipped pixels. Do not apply independent min-max normalization to each projection, because doing so breaks the common line-integral scale across views of the same object.

## File Format

Example `manifest.json`:

```json
{
  "format_version": 1,
  "scenes": [
    {
      "id": "case001",
      "group_id": "patient001",
      "domain": "medical",
      "split": "train",
      "projections": "case001.npz",
      "geometry": "case001_geometry.json",
      "volume": "case001_volume.npy",
      "input_index": 0,
      "target_indices": [1, 2, 3, 4]
    }
  ]
}
```

The NPZ key is `projections`, with shape `[V,H,W]` or `[V,1,H,W]`. Paths in the manifest are relative to the manifest directory. The volume file is optional and used only for evaluation; training does not load it. Each object requires at least one condition projection, one denoising supervision view, and another held-out supervision view.

`group_id` must identify an independent patient or physical object. Different acquisitions from the same group must not cross train/val/test partitions. Fix `input_index` and `target_indices` in advance; duplicate or overlapping indices cause an error. Training targets may be sampled randomly within the training set, while evaluation iterates over the predefined target views.

Example `geometry.json`:

```json
{
  "views": [
    {
      "c2w": [[0,0,-1,3],[1,0,0,0],[0,-1,0,0],[0,0,0,1]],
      "detector_size": [4.8,4.8],
      "dsd": 6.0,
      "beam": "cone",
      "offset": [0.0,0.0]
    }
  ],
  "canonical": {
    "c2w": [[0,0,-1,3],[1,0,0,0],[0,-1,0,0],[0,0,0,1]],
    "detector_size": [4.8,4.8],
    "dsd": 6.0,
    "beam": "cone",
    "offset": [0.0,0.0]
  }
}
```

This example illustrates a single camera object; the actual number of entries in `views` must equal V. The matrix transforms camera coordinates into global scanner coordinates. Its column axes are right/down/forward, with forward along local +z. `detector_size` is ordered as **[height,width]**, and `offset` as **[vertical,horizontal]**, using the same units as the global coordinates. Rays pass through pixel centers.

`canonical` specifies a predefined virtual target geometry. It determines the diffusion-canvas camera and contains no additional measurement. Calibration parameters may differ across scenes, but the canonical angle should be fixed in advance.

The inference camera file uses two keys, `condition` and `canonical`, each containing a camera object of the format above. Novel-view rendering reads the `views` list.

## Coordinate and Attenuation Scales

The main configuration uses a `[-1,1]^3` scene box. If physical coordinates are scaled as `x_new=s*x_physical`, preserve the line integral by using `mu_new=mu_physical/s`. Projection P is unchanged by this geometric unit conversion.

If projections are additionally normalized using a fixed scale S estimated only from training data, `P_model=P/S`, then the model's volume and kappa units are `mu_model=mu_physical/(s*S)`. The `projection_scale` setting is S; it must not be estimated from validation or test statistics. Projections exported by `infer.py` are multiplied back by S. Exported Gaussians and volumes remain in model units, with S recorded in metadata. Recover physical attenuation by multiplying by `s*S`, where s comes from the actual calibration.

Volume arrays use **xyz** axes and voxel-center sampling within the fixed scene box. Medical image arrays often use zyx ordering and must be aligned first. Resample volumes to the same cubic grid. The project does not automatically equate HU values with linear attenuation or silently assume DICOM orientation.

## Converting Existing Scanner Scenes

`scripts/convert_scanner.py` reads `scanner`, `proj_train`, and `proj_test` from `meta_data.json`. It accepts circular-trajectory calibration with `angle` expressed in radians. The original `proj_train` and `proj_test` entries describe acquisition views of the same scene; the catalog's scene/patient split determines the supervision partition in this project.

Example catalog:

```json
[
  {"id":"case001","path":"raw/case001","group_id":"object001","split":"train","domain":"industrial","input_index":0},
  {"id":"case002","path":"raw/case002","group_id":"object002","split":"val","domain":"industrial","input_index":0},
  {"id":"case003","path":"raw/case003","group_id":"object003","split":"test","domain":"industrial","input_index":0}
]
```

The converter does not move images from validation/test subjects into the training partition or automatically generate subject-ID lists claimed to match the manuscript. For nonzero `offOrigin`, a noncubic field of view, or a noncircular trajectory, supply verified complete geometry rather than relying on inferred calibration.

For LIDC-IDRI and Xt3D, provide appropriately obtained and calibrated/preprocessed data with the original splits. The bundled examples are synthetic phantoms. They do not include these real datasets or fabricated instance lists for the 808/101/101 or 534/67/67 splits.
