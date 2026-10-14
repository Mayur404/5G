"""Regular flat-top hexagonal simulation domains and boundary reflection."""

from __future__ import annotations

import math

import numpy as np


class HexDomain:
    def __init__(self, centers: np.ndarray, radius: float):
        self.centers = centers
        self.radius = radius
        angles = np.arange(6) * math.pi / 3
        offsets = radius * np.column_stack((np.cos(angles), np.sin(angles)))
        self.polygons = centers[:, None, :] + offsets
        angles = np.array([math.pi / 6, math.pi / 2, 5 * math.pi / 6])
        self.normals = np.column_stack((np.cos(angles), np.sin(angles)))
        self.apothem = radius * math.sqrt(3) / 2
        edges = {}
        for center, polygon in zip(centers, self.polygons, strict=True):
            for index, a in enumerate(polygon):
                b = polygon[(index + 1) % 6]
                key = tuple(sorted((tuple(np.round(a, 8)), tuple(np.round(b, 8)))))
                normal = (a + b) / 2 - center
                normal /= np.linalg.norm(normal)
                edges.setdefault(key, []).append((a, b, normal))
        self.boundary = [values[0] for values in edges.values() if len(values) == 1]

    def contains(self, position: np.ndarray) -> bool:
        projections = np.abs((position - self.centers) @ self.normals.T)
        return bool(np.any(np.all(projections <= self.apothem + 1e-8, axis=1)))

    def move(self, position: np.ndarray, velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Trace a full interval, reflecting only at exposed cluster edges.

        Shared edges are ordinary cell crossings, not reflecting walls. The
        unused displacement after a reflection is retained; speed is preserved.
        """
        start_inside = np.all(
            np.abs((position - self.centers) @ self.normals.T) <= self.apothem + 1e-8, axis=1
        )
        end = position + velocity
        end_inside = np.all(
            np.abs((end - self.centers) @ self.normals.T) <= self.apothem + 1e-8, axis=1
        )
        if np.any(start_inside & end_inside):
            # A segment contained in one convex tile cannot hit the outside
            # boundary. Do not use an endpoint-only shortcut across tiles:
            # the cluster's concave notches still require collision tracing.
            return end, velocity.copy()
        start = position.copy()
        point, remaining, reflected = position.copy(), velocity.copy(), velocity.copy()
        for _ in range(16):
            first = None
            for a, b, normal in self.boundary:
                outward = float(remaining @ normal)
                if outward <= 1e-12:
                    continue
                fraction = float((a - point) @ normal) / outward
                if fraction < -1e-9 or fraction > 1 + 1e-9:
                    continue
                hit = point + max(0.0, fraction) * remaining
                edge = b - a
                along = float((hit - a) @ edge) / float(edge @ edge)
                if -1e-9 <= along <= 1 + 1e-9 and (first is None or fraction < first[0]):
                    first = (max(0.0, min(1.0, fraction)), normal)
            if first is None:
                end = point + remaining
                if self.contains(end):
                    return end, reflected
                break
            fraction, normal = first
            point += fraction * remaining
            point -= normal * 1e-9  # avoid repeatedly re-hitting the same edge
            remaining *= 1 - fraction
            remaining -= 2 * float(remaining @ normal) * normal
            reflected -= 2 * float(reflected @ normal) * normal
            if np.linalg.norm(remaining) < 1e-12:
                return point, reflected
        # A deterministic safeguard for an exact multi-edge vertex collision.
        # Preserve an in-domain position and speed rather than leaking outside.
        return start, -velocity
