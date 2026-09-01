from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Cell:
    token: int
    vertices: tuple[tuple[float, ...], ...]
    min_gap: float

    @property
    def dimension(self) -> int:
        return len(self.vertices[0]) if self.vertices else 0


def _line_values(a: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:
    return a + b @ x


def _min_gap(a: np.ndarray, b: np.ndarray, token: int, vertices: np.ndarray) -> float:
    if len(vertices) == 0:
        return float("-inf")
    values = np.stack([_line_values(a, b, v) for v in vertices], axis=0)
    winner = values[:, token]
    competitors = np.delete(values, token, axis=1)
    return float(np.min(winner[:, None] - competitors))


def _as_cell(token: int, vertices: np.ndarray, min_gap: float) -> Cell:
    clean = tuple(tuple(float(x) for x in row) for row in vertices)
    return Cell(token=int(token), vertices=clean, min_gap=float(min_gap))


def partition_interval(a: np.ndarray, b: np.ndarray, tol: float = 1e-12) -> list[Cell]:
    """Exact argmax partition for logits l_v(s)=a_v+s b_v on [0,1]."""

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("a and b must have shape (vocab_size,)")

    breaks = {0.0, 1.0}
    vocab = len(a)
    for u in range(vocab):
        for v in range(u + 1, vocab):
            denom = b[u] - b[v]
            if abs(denom) <= tol:
                continue
            s = -(a[u] - a[v]) / denom
            if tol < s < 1.0 - tol:
                breaks.add(float(s))

    points = sorted(breaks)
    cells: list[Cell] = []
    for left, right in zip(points[:-1], points[1:]):
        if right - left <= tol:
            continue
        mid = 0.5 * (left + right)
        token = int(np.argmax(a + mid * b))
        vertices = np.array([[left], [right]], dtype=np.float64)
        gap = _min_gap(a, b[:, None], token, vertices)
        if cells and cells[-1].token == token and cells[-1].min_gap >= -tol:
            prev = np.array(cells[-1].vertices, dtype=np.float64)
            merged = np.array([[prev[0, 0]], [right]], dtype=np.float64)
            cells[-1] = _as_cell(token, merged, _min_gap(a, b[:, None], token, merged))
        else:
            cells.append(_as_cell(token, vertices, gap))
    return cells


def _polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _clip_halfplane(poly: np.ndarray, normal: np.ndarray, offset: float, tol: float) -> np.ndarray:
    """Clip polygon by normal @ x + offset >= 0."""

    if len(poly) == 0:
        return poly

    out: list[np.ndarray] = []
    prev = poly[-1]
    prev_val = float(normal @ prev + offset)
    prev_inside = prev_val >= -tol
    for curr in poly:
        curr_val = float(normal @ curr + offset)
        curr_inside = curr_val >= -tol
        if curr_inside != prev_inside:
            denom = prev_val - curr_val
            if abs(denom) > tol:
                t = prev_val / denom
                out.append(prev + t * (curr - prev))
        if curr_inside:
            out.append(curr)
        prev, prev_val, prev_inside = curr, curr_val, curr_inside

    if not out:
        return np.empty((0, 2), dtype=np.float64)
    return np.array(out, dtype=np.float64)


def partition_simplex(a: np.ndarray, b: np.ndarray, tol: float = 1e-10) -> list[Cell]:
    """Exact next-token partition on the 2-simplex.

    The logits have form l_v(lambda)=a_v + lambda_1 b_{v,1} + lambda_2 b_{v,2}.
    Each token cell is obtained by intersecting all halfplanes where that token
    beats every competitor.
    """

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if b.shape != (len(a), 2):
        raise ValueError("b must have shape (vocab_size, 2)")

    simplex = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    cells: list[Cell] = []
    vocab = len(a)
    for token in range(vocab):
        poly = simplex.copy()
        for competitor in range(vocab):
            if competitor == token:
                continue
            normal = b[token] - b[competitor]
            offset = float(a[token] - a[competitor])
            poly = _clip_halfplane(poly, normal, offset, tol)
            if len(poly) == 0:
                break
        if _polygon_area(poly) > tol:
            cells.append(_as_cell(token, poly, _min_gap(a, b, token, poly)))
    return cells


def certify_cells(cells: list[Cell], epsilon: float) -> list[Cell]:
    """Return whole cells whose minimum margin exceeds epsilon.

    This is a conservative summary helper. The paper's robust construction is
    implemented by ``robust_subregions``, which clips each cell by the
    epsilon-margin halfspaces.
    """

    return [cell for cell in cells if cell.min_gap > float(epsilon)]


def _robust_interval_cell(a: np.ndarray, b: np.ndarray, cell: Cell, epsilon: float, tol: float) -> Cell | None:
    token = int(cell.token)
    lo = float(cell.vertices[0][0])
    hi = float(cell.vertices[-1][0])
    slopes = np.asarray(b, dtype=np.float64).reshape(-1)
    for other in range(len(a)):
        if other == token:
            continue
        offset = float(a[token] - a[other] - epsilon)
        slope = float(slopes[token] - slopes[other])
        if abs(slope) <= tol:
            if offset <= tol:
                return None
            continue
        root = -offset / slope
        if slope > 0:
            lo = max(lo, root)
        else:
            hi = min(hi, root)
        if hi <= lo + tol:
            return None
    vertices = np.array([[lo], [hi]], dtype=np.float64)
    return _as_cell(token, vertices, _min_gap(a, slopes[:, None], token, vertices))


def _robust_simplex_cell(a: np.ndarray, b: np.ndarray, cell: Cell, epsilon: float, tol: float) -> Cell | None:
    token = int(cell.token)
    poly = np.array(cell.vertices, dtype=np.float64)
    coeffs = np.asarray(b, dtype=np.float64)
    for other in range(len(a)):
        if other == token:
            continue
        normal = coeffs[token] - coeffs[other]
        offset = float(a[token] - a[other] - epsilon)
        poly = _clip_halfplane(poly, normal, offset, tol)
        if _polygon_area(poly) <= tol:
            return None
    return _as_cell(token, poly, _min_gap(a, coeffs, token, poly))


def robust_subregions(cells: list[Cell], a: np.ndarray, b: np.ndarray, epsilon: float, tol: float = 1e-10) -> list[Cell]:
    """Clip greedy cells to epsilon-robust subregions.

    A returned subregion satisfies
    l_token(lambda) - l_other(lambda) > epsilon for every competitor at all
    vertices, which certifies the affine cell interior under a pairwise-gap
    error bound.
    """

    if not cells:
        return []
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    epsilon = float(epsilon)
    out: list[Cell] = []
    for cell in cells:
        if cell.dimension == 1:
            robust = _robust_interval_cell(a, b, cell, epsilon, tol)
        elif cell.dimension == 2:
            robust = _robust_simplex_cell(a, b, cell, epsilon, tol)
        else:
            raise ValueError("robust_subregions supports one- and two-dimensional cells")
        if robust is not None:
            out.append(robust)
    return out
