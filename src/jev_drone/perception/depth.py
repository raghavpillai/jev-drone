"""Fuse camera depth into observed travel clearance, without choosing controls."""
import numpy as np

from jev_drone.perception.sensors import basis, points


def observed_collision_distance(frames, origin, direction, radius):
    """First intersection of the moving body sphere with an observed depth point.

    Free-space samples alone can step over a narrow pole. Occupied pixels cap
    that estimate even when none of the nine footprint rays hits the obstacle.
    Points behind the center do not prevent moving away from an obstruction.
    """
    limit = float("inf")
    for frame in frames:
        if not hasattr(frame, "clearance_cloud"):
            cloud = points(frame)
            frame.clearance_cloud = cloud[np.isfinite(cloud).all(axis=2)]
        relative = frame.clearance_cloud-origin
        along = np.einsum("ij,j->i", relative, direction)
        perpendicular2 = np.einsum("ij,ij->i", relative, relative)-along*along
        hit = (along > 0) & (perpendicular2 < radius*radius) & (along < limit+radius)
        if hit.any():
            entry = along[hit]-np.sqrt(np.maximum(0., radius*radius-perpendicular2[hit]))
            limit = min(limit, max(0., float(entry.min())))
    return limit


def clearance(frames, origin, direction, radius=.40):
    direction, a, b = basis(np.array(direction, dtype=float, copy=True))
    occupied_limit = observed_collision_distance(frames, origin, direction, radius+.05)
    distances = np.array([.2, .4, .6, .9, 1.3, 1.8, 2.5, 3.5])
    footprint = np.array([radius*(x*a+y*b) for x in (-1, 0, 1) for y in (-1, 0, 1)])
    samples = (origin + (distances[:, None, None]+radius)*direction + footprint).reshape(-1, 3)
    observed = np.zeros(len(samples), dtype=bool)
    free = np.zeros(len(samples), dtype=bool)
    offsets = np.array([(y, x) for y in (-1, 0, 1) for x in (-1, 0, 1)])
    for frame in frames:
        height, width = frame.depth.shape
        relative = samples-frame.origin
        z = relative@frame.forward
        safe_z = np.maximum(z, .001)
        u = np.rint(width/2+frame.focal*(relative@frame.right)/safe_z).astype(int)
        v = np.rint(height/2-frame.focal*(relative@frame.up)/safe_z).astype(int)
        valid = (z > .06) & (u >= 1) & (u < width-1) & (v >= 1) & (v < height-1)
        indices = np.flatnonzero(valid)
        patch = frame.depth[v[indices, None]+offsets[:, 0], u[indices, None]+offsets[:, 1]]
        finite = np.isfinite(patch)
        count = finite.sum(axis=1)
        ordered = np.sort(np.where(finite, patch, np.inf), axis=1)
        # Quantile interpolation over finite values only, including partial dropout.
        rank = np.maximum(count-1, 0)*.2
        lo, hi = np.floor(rank).astype(int), np.ceil(rank).astype(int)
        rows = np.arange(len(indices))
        depth = ordered[rows, lo].copy()
        interpolate = hi != lo
        depth[interpolate] += (ordered[rows[interpolate], hi[interpolate]]-depth[interpolate])*(rank[interpolate]-lo[interpolate])
        seen = count >= 3
        observed[indices] |= seen
        free[indices] |= seen & (depth >= z[indices]+.035)
    seen_tubes = observed.reshape(-1, 9).all(axis=1)
    free_tubes = free.reshape(-1, 9).all(axis=1)
    last = 0.
    for distance, seen, clear in zip(distances, seen_tubes, free_tubes):
        if not clear:
            return round(min(last, occupied_limit), 3) if last or seen else None
        last = float(distance)
    return round(min(last, occupied_limit), 3)
