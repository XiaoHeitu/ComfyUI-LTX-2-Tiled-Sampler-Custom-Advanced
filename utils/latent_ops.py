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


def _slice_video_spatial(tensor: torch.Tensor, v_start: int, v_end: int, h_start: int, h_end: int) -> torch.Tensor:
    return tensor[:, :, :, v_start:v_end, h_start:h_end]


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
