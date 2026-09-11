"""Optional evaluation-only region and binary structure metrics.

NSD here uses equally weighted boundary voxels; it is explicitly distinguished
from surfel-area-weighted implementations. Use the same definition for controls.
"""
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def dice(prediction, truth):
    p, t = np.asarray(prediction, bool), np.asarray(truth, bool)
    denominator = p.sum()+t.sum()
    return 1.0 if denominator == 0 else float(2*(p&t).sum()/denominator)


def normalized_surface_dice(prediction, truth, spacing=(1., 1., 1.), tolerance=1.):
    p, t = np.asarray(prediction, bool), np.asarray(truth, bool)
    if not p.any() or not t.any():
        return float(not p.any() and not t.any())
    ps = p & ~binary_erosion(p)
    ts = t & ~binary_erosion(t)
    distance_to_p = distance_transform_edt(~ps, sampling=spacing)
    distance_to_t = distance_transform_edt(~ts, sampling=spacing)
    return float(((distance_to_t[ps] <= tolerance).sum()+(distance_to_p[ts] <= tolerance).sum())/(ps.sum()+ts.sum()))


def cldice(prediction, truth):
    from skimage.morphology import skeletonize
    p, t = np.asarray(prediction, bool), np.asarray(truth, bool)
    if not p.any() or not t.any():
        return float(not p.any() and not t.any())
    sp, st = skeletonize(p).astype(bool), skeletonize(t).astype(bool)
    precision = float((sp&t).sum()/max(sp.sum(), 1))
    sensitivity = float((st&p).sum()/max(st.sum(), 1))
    return 2*precision*sensitivity/max(precision+sensitivity, 1e-12)


def masked_reconstruction(prediction, truth, mask, data_range=1.):
    mask = np.asarray(mask, bool)
    if not mask.any():
        raise ValueError("Evaluation ROI is empty")
    errors = np.asarray(prediction)[mask]-np.asarray(truth)[mask]
    return {"masked_psnr": float(10*np.log10(data_range**2/max(np.mean(errors**2), 1e-12))), "masked_mae": float(np.mean(np.abs(errors)))}
