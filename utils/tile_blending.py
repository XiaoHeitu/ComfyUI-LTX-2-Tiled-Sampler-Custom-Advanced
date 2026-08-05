from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TileRegion:
    row: int
    col: int
    v_start: int
    v_end: int
    h_start: int
    h_end: int
    top_overlap: int
    bottom_overlap: int
    left_overlap: int
    right_overlap: int

    @property
    def height(self) -> int:
        return self.v_end - self.v_start

    @property
    def width(self) -> int:
        return self.h_end - self.h_start


def _build_axis_positions(total: int, tile: int, overlap: int) -> list[tuple[int, int]]:
    if tile <= 1 or tile >= total:
        return [(0, total)]

    step = max(1, tile - overlap)
    positions: list[tuple[int, int]] = []
    start = 0

    while True:
        end = min(total, start + tile)
        positions.append((start, end))
        if end >= total:
            break

        next_start = start + step
        last_start = max(0, total - tile)
        if next_start > last_start:
            next_start = last_start
        if positions and next_start == positions[-1][0]:
            break
        start = next_start

    return positions


def build_spatial_tiles(height: int, width: int, tile: int, overlap: int) -> list[TileRegion]:
    v_positions = _build_axis_positions(height, tile, overlap)
    h_positions = _build_axis_positions(width, tile, overlap)

    regions: list[TileRegion] = []
    for row, (v_start, v_end) in enumerate(v_positions):
        for col, (h_start, h_end) in enumerate(h_positions):
            top_overlap = v_positions[row - 1][1] - v_start if row > 0 else 0
            bottom_overlap = v_end - v_positions[row + 1][0] if row < len(v_positions) - 1 else 0
            left_overlap = h_positions[col - 1][1] - h_start if col > 0 else 0
            right_overlap = h_end - h_positions[col + 1][0] if col < len(h_positions) - 1 else 0
            regions.append(
                TileRegion(
                    row=row,
                    col=col,
                    v_start=v_start,
                    v_end=v_end,
                    h_start=h_start,
                    h_end=h_end,
                    top_overlap=max(0, top_overlap),
                    bottom_overlap=max(0, bottom_overlap),
                    left_overlap=max(0, left_overlap),
                    right_overlap=max(0, right_overlap),
                )
            )
    return regions


def _build_weight_mask(tile_tensor: torch.Tensor, region: TileRegion) -> torch.Tensor:
    weights = torch.ones_like(tile_tensor)

    if region.left_overlap > 0:
        left = min(region.left_overlap, region.width)
        ramp = torch.linspace(0, 1, left, device=tile_tensor.device, dtype=tile_tensor.dtype)
        weights[:, :, :, :, :left] *= ramp.view(1, 1, 1, 1, -1)
    if region.right_overlap > 0:
        right = min(region.right_overlap, region.width)
        ramp = torch.linspace(1, 0, right, device=tile_tensor.device, dtype=tile_tensor.dtype)
        weights[:, :, :, :, -right:] *= ramp.view(1, 1, 1, 1, -1)
    if region.top_overlap > 0:
        top = min(region.top_overlap, region.height)
        ramp = torch.linspace(0, 1, top, device=tile_tensor.device, dtype=tile_tensor.dtype)
        weights[:, :, :, :top, :] *= ramp.view(1, 1, 1, -1, 1)
    if region.bottom_overlap > 0:
        bottom = min(region.bottom_overlap, region.height)
        ramp = torch.linspace(1, 0, bottom, device=tile_tensor.device, dtype=tile_tensor.dtype)
        weights[:, :, :, -bottom:, :] *= ramp.view(1, 1, 1, -1, 1)

    return weights


def blend_tiles(base_shape: tuple[int, ...], tile_outputs: list[tuple[TileRegion, torch.Tensor]]) -> torch.Tensor:
    if not tile_outputs:
        raise ValueError("没有可融合的 tile 输出。")

    sample = tile_outputs[0][1]
    output = torch.zeros(base_shape, device=sample.device, dtype=sample.dtype)
    weights = torch.zeros_like(output)

    for region, tile_tensor in tile_outputs:
        weight_mask = _build_weight_mask(tile_tensor, region)
        output[:, :, :, region.v_start:region.v_end, region.h_start:region.h_end] += tile_tensor * weight_mask
        weights[:, :, :, region.v_start:region.v_end, region.h_start:region.h_end] += weight_mask

    return output / torch.clamp(weights, min=1e-8)
