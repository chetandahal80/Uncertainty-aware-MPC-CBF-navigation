"""Current-frame metric RGB-D measurements, independent of image-filter lag."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def depth_mode(depth_image, bbox, minimum=0.15, maximum=9.9, noise_floor=0.1):
    """Return a supported native-float depth mode and robust depth spread.

    Keep sensor invalid values out of the histogram. The spread is not divided
    by sqrt(pixel_count): body shape and depth errors are spatially correlated.
    """
    image = np.asarray(depth_image)
    if image.ndim != 2:
        raise ValueError("Depth image must be two dimensional")
    box = np.asarray(bbox[:4], dtype=float)
    if not np.all(np.isfinite(box)):
        return None
    x1, y1 = np.floor(box[:2]).astype(int)
    x2, y2 = np.ceil(box[2:]).astype(int)
    x1, x2 = np.clip([x1, x2], 0, image.shape[1])
    y1, y2 = np.clip([y1, y2], 0, image.shape[0])
    if x2 <= x1 or y2 <= y1:
        return None
    roi = image[y1:y2, x1:x2]
    valid = roi[np.isfinite(roi) & (roi > minimum) & (roi < maximum)]
    if valid.size < 6:
        return None
    counts, edges = np.histogram(valid, bins=np.arange(minimum, maximum + 0.05, 0.05))
    smooth = np.convolve(counts, [0.25, 0.5, 0.25], mode="same")
    index = int(np.argmax(smooth))
    centre = (edges[index] + edges[index + 1]) / 2
    cluster = valid[np.abs(valid - centre) < 0.12]
    if cluster.size < 6 or cluster.size < 0.03 * valid.size:
        return None
    # Select a supported nearer layer when the central region and full-box
    # dominant mode are separated by more than 0.5 m.
    width = x2 - x1
    height = y2 - y1
    seed = image[
        y1 + int(0.10 * height) : y1 + int(0.65 * height),
        x1 + int(0.30 * width) : x1 + int(0.70 * width),
    ]
    seed = seed[np.isfinite(seed) & (seed > minimum) & (seed < maximum)]
    if seed.size >= 6:
        seed_counts, _ = np.histogram(seed, bins=edges)
        seed_counts = np.convolve(seed_counts, [0.25, 0.5, 0.25], mode="same")
        j = int(np.argmax(seed_counts))
        seed_centre = (edges[j] + edges[j + 1]) / 2
        seed_cluster = seed[np.abs(seed - seed_centre) < 0.12]
        if seed_cluster.size >= max(6, 0.2 * seed.size) and centre - seed_centre > 0.5:
            alternative = valid[np.abs(valid - seed_centre) < 0.12]
            if alternative.size >= max(6, 0.03 * valid.size):
                cluster = alternative
    depth = float(np.median(cluster))
    mad = float(np.median(np.abs(cluster - depth)))
    return depth, max(float(noise_floor), 1.4826 * mad), int(cluster.size)


def current_frame_rows(detections, filtered_rows, pixel_std, depth_std):
    """Use image filters for association, never their delayed mean as a sensor.

    Fresh filtered rows are matched one-to-one to current detections. The world
    filter receives the actual current bbox/depth and measurement covariance,
    avoiding a second correction with correlated image-filter posterior noise.
    """
    detections = np.asarray(detections, dtype=float)
    filtered = np.asarray(filtered_rows, dtype=float)
    if detections.size == 0 or filtered.size == 0:
        return []
    detections = detections.reshape(-1, 6)
    filtered = filtered.reshape(-1, 6)
    valid = (
        np.all(np.isfinite(detections), axis=1)
        & (detections[:, 2] > detections[:, 0])
        & (detections[:, 3] > detections[:, 1])
        & (detections[:, 4] > 0.15)
        & (detections[:, 4] < 9.9)
    )
    detections = detections[valid]
    if not len(detections):
        return []
    dc = (detections[:, :2] + detections[:, 2:4]) / 2
    fc = (filtered[:, :2] + filtered[:, 2:4]) / 2
    scale = np.maximum(
        np.linalg.norm(detections[:, 2:4] - detections[:, :2], axis=1), 1.0
    )
    cost = np.linalg.norm(dc[:, None, :] - fc[None, :, :], axis=2) / scale[
        :, None
    ] + 0.2 * np.abs(detections[:, None, 4] - filtered[None, :, 4])
    rows = []
    for i, j in zip(*linear_sum_assignment(cost)):
        d = detections[i]
        rows.append(
            dict(
                bbox=d[:4].tolist(),
                depth=float(d[4]),
                id=int(filtered[j, 5]),
                covariance_uvd=np.diag(
                    [pixel_std**2, pixel_std**2, depth_std**2]
                ).tolist(),
            )
        )
    return rows
