"""Presentation smoothing for one-second motion scores; raw scores stay unchanged.

Supply neighboring scores across chunk boundaries, smooth, then trim context.
None/NaN marks a gap and is preserved. At true run boundaries, repeat the edge
score instead of inserting zeros. Weights operate over seconds, not video frames.
"""
import numpy as np

GENTLE = (1, 2, 1)
STRONGER = (1, 2, 3, 2, 1)


def smooth_scores(scores, weights=GENTLE):
    values = np.asarray(scores, dtype=float)
    kernel = np.asarray(weights, dtype=float)
    if values.ndim != 1 or kernel.ndim != 1:
        raise ValueError('Scores and weights must be one-dimensional')
    if not len(kernel) or len(kernel) % 2 != 1:
        raise ValueError('Weights must have a positive odd length')
    if not np.all(np.isfinite(kernel)) or np.any(kernel < 0) or kernel.sum() <= 0:
        raise ValueError('Weights must be finite, nonnegative, with positive sum')
    if np.any(np.isinf(values)):
        raise ValueError('Use None or NaN for gaps, not infinity')
    valid = np.isfinite(values)
    if np.any((values[valid] < 0) | (values[valid] > 1)):
        raise ValueError('Motion scores must be between zero and one')
    result = np.full(values.shape, np.nan)
    kernel = kernel / kernel.sum()
    radius = len(kernel) // 2
    # Process each continuous run independently, preserving missing coverage.
    edges = np.diff(np.r_[False, valid, False].astype(int))
    for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        padded = np.pad(values[start:end], (radius, radius), mode='edge')
        result[start:end] = np.convolve(padded, kernel, mode='valid')
    return result
