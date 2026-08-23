from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import comfy.nested_tensor


@dataclass(frozen=True)
class TimeWindow:
    index: int
    history_start: int
    end: int
    retain_start: int
    retain_end: int

    @property
    def length(self) -> int:
        return self.end - self.history_start

    @property
    def retain_length(self) -> int:
        return self.retain_end - self.retain_start

    @property
    def is_first(self) -> bool:
        return self.index == 0 and self.retain_start == 0


def detect_latent_kind(samples) -> str:
    if getattr(samples, "is_nested", False):
        tensors = samples.unbind()
        if len(tensors) == 2:
            return "ltxav"
        raise ValueError("当前节点仅支持两分支 NestedTensor（video + audio）的 LTXAV latent。")
    if isinstance(samples, torch.Tensor) and samples.ndim == 5:
        return "ltxv"
    raise ValueError("当前节点仅支持 LTX 视频 latent 或 LTXAV latent。")


def build_time_windows(total_frames: int, add_frames: int, overlap_frames: int) -> list[TimeWindow]:
    if total_frames < 1:
        raise ValueError("latent 时间长度必须 >= 1。")
    if add_frames < 1:
        raise ValueError("采样帧数必须 >= 1。")

    window_frames = add_frames + overlap_frames
    first_end = min(total_frames, window_frames)
    windows = [TimeWindow(index=0, history_start=0, end=first_end, retain_start=0, retain_end=first_end)]

    cursor = first_end
    index = 1
    while cursor < total_frames:
        history_start = max(0, cursor - overlap_frames)
        end = min(total_frames, cursor + add_frames)
        retain_start = cursor - history_start
        retain_end = retain_start + (end - cursor)
        windows.append(
            TimeWindow(
                index=index,
                history_start=history_start,
                end=end,
                retain_start=retain_start,
                retain_end=retain_end,
            )
        )
        cursor = end
        index += 1

    return windows


def move_to_device(value, device):
    if value is None:
        return None
    if getattr(value, "is_nested", False):
        return value.to(device=device)
    return value.to(device=device)


def _slice_time(tensor: torch.Tensor, start: int, end: int, dim: int = 2) -> torch.Tensor:
    index = [slice(None)] * tensor.ndim
    index[dim] = slice(start, end)
    return tensor[tuple(index)]


def prepend_first_temporal_slice(tensor: torch.Tensor | None, dim: int = 2):
    if tensor is None:
        return None
    first = _slice_time(tensor, 0, 1, dim=dim)
    return torch.cat([first, tensor], dim=dim)


def strip_first_temporal_slice(tensor: torch.Tensor | None, dim: int = 2):
    if tensor is None:
        return None
    if tensor.shape[dim] <= 1:
        raise ValueError("无法移除时间首帧，占位帧长度不足。")
    return _slice_time(tensor, 1, tensor.shape[dim], dim=dim)


def _make_temporal_mask_like(reference: torch.Tensor, dim: int = 2) -> torch.Tensor:
    shape = list(reference.shape)
    if len(shape) < 3:
        raise ValueError("时间 mask 至少需要 batch/channel/time 三个维度。")
    shape[1] = 1
    for index in range(3, len(shape)):
        shape[index] = 1
    return torch.ones(tuple(shape), dtype=torch.float32, device=reference.device)


def freeze_temporal_history(reference: torch.Tensor, base_mask: torch.Tensor | None, history_frames: int, dim: int = 2):
    if history_frames <= 0:
        return base_mask

    mask = base_mask.clone() if base_mask is not None else _make_temporal_mask_like(reference, dim=dim)
    frozen = max(0, min(int(history_frames), mask.shape[dim]))
    if frozen <= 0:
        return mask

    index = [slice(None)] * mask.ndim
    index[dim] = slice(0, frozen)
    mask[tuple(index)] = 0
    return mask


def replace_time_slice(tensor: torch.Tensor, start: int, end: int, value: torch.Tensor, dim: int = 2) -> torch.Tensor:
    updated = tensor.clone()
    index = [slice(None)] * updated.ndim
    index[dim] = slice(start, end)
    updated[tuple(index)] = value
    return updated


def _slice_video_spatial(tensor: torch.Tensor, v_start: int, v_end: int, h_start: int, h_end: int) -> torch.Tensor:
    if tensor.ndim < 5:
        return tensor

    height = tensor.shape[3]
    width = tensor.shape[4]

    # Some LTX noise masks are temporal-only placeholders with singleton
    # spatial dims (for example 1x1). Those masks should keep their full
    # spatial extent instead of being cropped by video tile coordinates.
    if height <= 1:
        v_slice = slice(0, height)
    else:
        v0 = max(0, min(v_start, height))
        v1 = max(v0, min(v_end, height))
        v_slice = slice(v0, v1)

    if width <= 1:
        h_slice = slice(0, width)
    else:
        h0 = max(0, min(h_start, width))
        h1 = max(h0, min(h_end, width))
        h_slice = slice(h0, h1)

    return tensor[:, :, :, v_slice, h_slice]


def split_av_components(value):
    if value is None:
        return None, None
    if not getattr(value, "is_nested", False):
        raise ValueError("预期为 NestedTensor，但收到普通 tensor。")
    tensors = value.unbind()
    if len(tensors) != 2:
        raise ValueError("当前节点仅支持 video + audio 两分支的 NestedTensor。")
    return tensors[0], tensors[1]


def map_video_boundary_to_audio(index: int, video_total: int, audio_total: int, mode: str) -> int:
    if video_total <= 0:
        return 0
    scaled = index * audio_total / video_total
    if mode == "start":
        value = math.floor(scaled)
    elif mode == "end":
        value = math.ceil(scaled)
    else:
        raise ValueError(f"未知 boundary mode: {mode}")
    return max(0, min(audio_total, value))


def slice_video_window_payload(samples, noise, noise_mask, window: TimeWindow):
    return (
        _slice_time(samples, window.history_start, window.end, dim=2),
        _slice_time(noise, window.history_start, window.end, dim=2),
        _slice_time(noise_mask, window.history_start, window.end, dim=2) if noise_mask is not None else None,
    )


def build_av_window_payload(samples, noise, noise_mask, window: TimeWindow):
    video_samples, audio_samples = split_av_components(samples)
    video_noise, audio_noise = split_av_components(noise)
    video_mask, audio_mask = split_av_components(noise_mask) if noise_mask is not None else (None, None)

    v_total = video_samples.shape[2]
    a_total = audio_samples.shape[2]

    a_history_start = map_video_boundary_to_audio(window.history_start, v_total, a_total, "start")
    a_end = map_video_boundary_to_audio(window.end, v_total, a_total, "end")
    a_added_start = map_video_boundary_to_audio(window.history_start + window.retain_start, v_total, a_total, "start")
    a_added_end = map_video_boundary_to_audio(window.history_start + window.retain_end, v_total, a_total, "end")

    if a_end <= a_history_start:
        a_end = min(a_total, a_history_start + 1)
    if a_added_start < a_history_start:
        a_added_start = a_history_start
    if a_added_end < a_added_start:
        a_added_end = a_added_start

    audio_retain_start = a_added_start - a_history_start
    audio_retain_end = audio_retain_start + (a_added_end - a_added_start)

    return {
        "video_history_start": window.history_start,
        "video_end": window.end,
        "audio_history_start": a_history_start,
        "audio_end": a_end,
        "video_samples": _slice_time(video_samples, window.history_start, window.end, dim=2),
        "audio_samples": _slice_time(audio_samples, a_history_start, a_end, dim=2),
        "video_noise": _slice_time(video_noise, window.history_start, window.end, dim=2),
        "audio_noise": _slice_time(audio_noise, a_history_start, a_end, dim=2),
        "video_mask": _slice_time(video_mask, window.history_start, window.end, dim=2) if video_mask is not None else None,
        "audio_mask": _slice_time(audio_mask, a_history_start, a_end, dim=2) if audio_mask is not None else None,
        "audio_retain_start": audio_retain_start,
        "audio_retain_end": audio_retain_end,
    }


def slice_video_tile(tensor: torch.Tensor, region) -> torch.Tensor:
    return _slice_video_spatial(tensor, region.v_start, region.v_end, region.h_start, region.h_end)


def make_nested(video_tensor: torch.Tensor, audio_tensor: torch.Tensor):
    return comfy.nested_tensor.NestedTensor((video_tensor, audio_tensor))


def update_video_window_context(samples: torch.Tensor, window: TimeWindow, chunk_samples: torch.Tensor) -> torch.Tensor:
    return replace_time_slice(samples, window.history_start, window.end, chunk_samples, dim=2)


def update_av_window_context(samples, payload: dict, chunk_samples):
    video_samples, audio_samples = split_av_components(samples)
    chunk_video, chunk_audio = split_av_components(chunk_samples)
    video_updated = replace_time_slice(
        video_samples,
        payload["video_history_start"],
        payload["video_end"],
        chunk_video,
        dim=2,
    )
    audio_updated = replace_time_slice(
        audio_samples,
        payload["audio_history_start"],
        payload["audio_end"],
        chunk_audio,
        dim=2,
    )
    return make_nested(video_updated, audio_updated)
