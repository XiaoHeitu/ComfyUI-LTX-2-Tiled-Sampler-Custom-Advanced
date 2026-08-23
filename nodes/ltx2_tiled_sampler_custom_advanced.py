from __future__ import annotations

from comfy_api.latest import io
import torch
import comfy.model_management
import comfy.sample
import comfy.utils
import comfy.nested_tensor
import latent_preview
import nodes as comfy_nodes
from tqdm import tqdm

from ..utils.latent_ops import (
    build_av_window_payload,
    build_time_windows,
    detect_latent_kind,
    freeze_temporal_history,
    make_nested,
    move_to_device,
    prepend_first_temporal_slice,
    slice_video_tile,
    slice_video_window_payload,
    split_av_components,
    strip_first_temporal_slice,
    update_av_window_context,
    update_video_window_context,
)
from ..utils.tile_blending import blend_tiles, build_spatial_tiles


class _GlobalSamplerProgress:
    def __init__(self, model_patcher, total_steps: int, node_id: str | None = None):
        self.total_steps = max(1, int(total_steps))
        self.current_step = 0
        self.preview_format = "JPEG"
        self.progress_bar = comfy.utils.ProgressBar(self.total_steps, node_id=node_id)
        self.console_progress = tqdm(total=self.total_steps, desc="LTX2TiledSampler", unit="steps", leave=True)
        self.previewer = latent_preview.get_previewer(model_patcher.load_device, model_patcher.model.latent_format)

    def reserve_subsample(self, local_steps: int, count_towards_total: bool = True) -> tuple[int, int]:
        local_steps = max(0, int(local_steps))
        start_step = self.current_step
        if count_towards_total:
            self.current_step += local_steps
        return start_step, local_steps

    def build_callback(self, start_step: int, local_steps: int, x0_output_dict=None, count_towards_total: bool = True):
        def callback(step, x0, x, total_steps):
            if x0_output_dict is not None:
                x0_output_dict["x0"] = x0

            preview_bytes = None
            if self.previewer is not None:
                preview_source = x0.tensors[0] if getattr(x0, "is_nested", False) else x0
                preview_bytes = self.previewer.decode_latent_to_preview_image(self.preview_format, preview_source)

            if count_towards_total and local_steps > 0:
                self.update_absolute(start_step + step + 1, preview_bytes)
            elif preview_bytes is not None:
                self.progress_bar.update_absolute(start_step, self.total_steps, preview_bytes)

        return callback

    def update_absolute(self, value: int, preview_bytes=None):
        value = max(0, min(int(value), self.total_steps))
        self.progress_bar.update_absolute(value, self.total_steps, preview_bytes)
        if self.console_progress is not None:
            delta = value - self.console_progress.n
            if delta > 0:
                self.console_progress.update(delta)

    def close(self):
        if self.console_progress is not None:
            remaining = self.total_steps - self.console_progress.n
            if remaining > 0:
                self.console_progress.update(remaining)
            self.console_progress.close()
            self.console_progress = None


class LTX2TiledSamplerCustomAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LTX2TiledSamplerCustomAdvanced",
            display_name="LTX2自定义采样器(分片)",
            category="model/sampling/custom",
            inputs=[
                io.Noise.Input("noise"),
                io.Guider.Input("guider"),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),
                io.Latent.Input("latent_image"),
                io.Int.Input(
                    "sample_frames",
                    default=32,
                    min=1,
                    max=4096,
                    tooltip="每个后续时间窗口新增的 latent 帧数。",
                ),
                io.Int.Input(
                    "overlap_frames",
                    default=4,
                    min=0,
                    max=4096,
                    tooltip="时间窗口之间回看的 latent 重叠帧数。",
                ),
                io.Int.Input(
                    "tile_size",
                    default=512,
                    min=32,
                    max=comfy_nodes.MAX_RESOLUTION,
                    step=8,
                    tooltip="空间分片尺寸，使用像素单位；内部会按 LTX latent 缩放比例转换为 latent 空间 tile。",
                ),
                io.Int.Input(
                    "tile_overlap",
                    default=192,
                    min=0,
                    max=comfy_nodes.MAX_RESOLUTION,
                    step=8,
                    tooltip="空间分片重合宽度，使用像素单位；内部会按 LTX latent 缩放比例转换为 latent 空间 overlap。",
                ),
            ],
            outputs=[
                io.Latent.Output(display_name="output"),
                io.Latent.Output(display_name="denoised_output"),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def _get_spatial_ratio(cls, latent: dict, guider) -> int:
        ratio = latent.get("downscale_ratio_spacial", None)
        if ratio is not None:
            try:
                return max(1, int(round(float(ratio))))
            except (TypeError, ValueError):
                pass
        latent_format = guider.model_patcher.get_model_object("latent_format")
        return max(1, int(getattr(latent_format, "spacial_downscale_ratio", 32)))

    @classmethod
    def _convert_pixel_tiles_to_latent(cls, tile_size: int, tile_overlap: int, spatial_ratio: int) -> tuple[int, int]:
        latent_tile = max(1, int(round(tile_size / spatial_ratio)))
        latent_overlap = max(0, int(round(tile_overlap / spatial_ratio)))
        if latent_overlap >= latent_tile:
            raise ValueError(
                f"重合像素转换到 latent 空间后不能大于等于分片大小：tile={latent_tile}, overlap={latent_overlap}。"
            )
        return latent_tile, latent_overlap

    @classmethod
    def _postprocess_x0(cls, guider, raw_x0, sampled):
        x0 = raw_x0
        if getattr(sampled, "is_nested", False) and not getattr(x0, "is_nested", False):
            latent_shapes = [x.shape for x in sampled.unbind()]
            x0 = comfy.nested_tensor.NestedTensor(comfy.utils.unpack_latents(x0, latent_shapes))
        x0_out = guider.model_patcher.model.process_latent_out(x0.cpu())
        return move_to_device(x0_out, comfy.model_management.intermediate_device())

    @classmethod
    def _normalize_output_structure(cls, value, latent_kind: str, reference_samples):
        if value is None:
            return None
        if latent_kind != "ltxav":
            return value
        if getattr(value, "is_nested", False):
            tensors = value.unbind()
            if len(tensors) == 2:
                return value
            raise ValueError(f"LTXAV 输出必须包含 2 个分支，当前得到 {len(tensors)} 个。")

        ref_video, ref_audio = split_av_components(reference_samples)
        rebuilt = comfy.utils.unpack_latents(value, [ref_video.shape, ref_audio.shape])
        if len(rebuilt) != 2:
            raise ValueError(f"LTXAV 输出重建失败，期望 2 个分支，当前得到 {len(rebuilt)} 个。")
        print("[LTX2TiledSampler] 检测到 AV 输出为普通 tensor，已按原始分支形状重建为 NestedTensor。", flush=True)
        return make_nested(rebuilt[0], rebuilt[1])

    @classmethod
    def _run_subsample(
        cls,
        guider,
        sampler,
        sigmas,
        latent_samples,
        sub_noise,
        sub_mask,
        seed,
        progress_state: _GlobalSamplerProgress | None = None,
        count_towards_progress: bool = True,
    ):
        x0_output = {}
        local_steps = max(0, int(sigmas.shape[-1] - 1))
        if progress_state is None:
            callback = latent_preview.prepare_callback(guider.model_patcher, local_steps, x0_output)
        else:
            start_step, reserved_steps = progress_state.reserve_subsample(local_steps, count_towards_total=count_towards_progress)
            callback = progress_state.build_callback(
                start_step,
                reserved_steps,
                x0_output_dict=x0_output,
                count_towards_total=count_towards_progress,
            )
        # When we aggregate progress across many sub-samples, keep the sampler's
        # internal tqdm disabled so the console shows a single global progress bar.
        disable_pbar = True if progress_state is not None else not comfy.utils.PROGRESS_BAR_ENABLED
        sampled = guider.sample(
            sub_noise,
            latent_samples,
            sampler,
            sigmas,
            denoise_mask=sub_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )
        sampled = move_to_device(sampled, comfy.model_management.intermediate_device())
        x0 = cls._postprocess_x0(guider, x0_output["x0"], sampled) if "x0" in x0_output else None
        return sampled, x0

    @classmethod
    def _make_video_active_mask(cls, tile_video_samples, tile_video_mask):
        if tile_video_mask is not None:
            return tile_video_mask
        return torch.ones(
            (tile_video_samples.shape[0], 1, tile_video_samples.shape[2], tile_video_samples.shape[3], tile_video_samples.shape[4]),
            dtype=torch.float32,
            device=tile_video_samples.device,
        )

    @classmethod
    def _make_frozen_audio_mask(cls, audio_latent):
        return torch.zeros(
            (audio_latent.shape[0], 1, audio_latent.shape[2], 1),
            dtype=torch.float32,
            device=audio_latent.device,
        )

    @classmethod
    def _prime_window_audio(
        cls,
        guider,
        sampler,
        sigmas,
        video_samples,
        video_noise,
        video_mask,
        audio_samples,
        audio_noise,
        audio_mask,
        region,
        seed,
        progress_state: _GlobalSamplerProgress | None = None,
    ):
        tile_video_samples = slice_video_tile(video_samples, region)
        tile_video_noise = slice_video_tile(video_noise, region)
        tile_video_mask = slice_video_tile(video_mask, region) if video_mask is not None else None
        nested_samples = make_nested(tile_video_samples, audio_samples)
        nested_noise = make_nested(tile_video_noise, audio_noise)
        nested_mask = make_nested(tile_video_mask, audio_mask) if (tile_video_mask is not None or audio_mask is not None) else None
        print(
            f"[LTX2TiledSampler] 使用首个 tile 预生成当前时间窗口的冻结音频 "
            f"row={region.row} col={region.col}",
            flush=True,
        )
        sampled, x0 = cls._run_subsample(
            guider,
            sampler,
            sigmas,
            nested_samples,
            nested_noise,
            nested_mask,
            seed,
            progress_state=progress_state,
            count_towards_progress=False,
        )
        _, sampled_audio = split_av_components(sampled)
        x0_audio = None
        if x0 is not None:
            _, x0_audio = split_av_components(x0)
        return sampled_audio, x0_audio

    @classmethod
    def _sample_video_window(
        cls,
        guider,
        sampler,
        sigmas,
        window_samples,
        window_noise,
        window_mask,
        latent_tile,
        latent_overlap,
        seed,
        causal_fix=False,
        progress_state: _GlobalSamplerProgress | None = None,
    ):
        base_shape = tuple(window_samples.shape)
        if causal_fix:
            print("[LTX2TiledSampler] 对非首个视频时间窗口应用 LTX 首帧因果补偿。", flush=True)
            window_samples = prepend_first_temporal_slice(window_samples, dim=2)
            window_noise = prepend_first_temporal_slice(window_noise, dim=2)
            window_mask = prepend_first_temporal_slice(window_mask, dim=2) if window_mask is not None else None

        height = window_samples.shape[3]
        width = window_samples.shape[4]
        regions = build_spatial_tiles(height, width, latent_tile, latent_overlap)

        tile_outputs = []
        tile_x0_outputs = []

        for region in regions:
            print(
                f"[LTX2TiledSampler] 时间窗口内空间 tile row={region.row} col={region.col} "
                f"v=({region.v_start}:{region.v_end}) h=({region.h_start}:{region.h_end})",
                flush=True,
            )
            tile_samples = slice_video_tile(window_samples, region)
            tile_noise = slice_video_tile(window_noise, region)
            tile_mask = slice_video_tile(window_mask, region) if window_mask is not None else None
            sampled, x0 = cls._run_subsample(
                guider,
                sampler,
                sigmas,
                tile_samples,
                tile_noise,
                tile_mask,
                seed,
                progress_state=progress_state,
            )
            if causal_fix:
                sampled = strip_first_temporal_slice(sampled, dim=2)
                if x0 is not None:
                    x0 = strip_first_temporal_slice(x0, dim=2)
            tile_outputs.append((region, sampled))
            if x0 is not None:
                tile_x0_outputs.append((region, x0))

        blended = blend_tiles(
            (base_shape[0], base_shape[1], base_shape[2], base_shape[3], base_shape[4]),
            tile_outputs,
        )
        blended_x0 = None
        if tile_x0_outputs and len(tile_x0_outputs) == len(tile_outputs):
            blended_x0 = blend_tiles(
                (base_shape[0], base_shape[1], base_shape[2], base_shape[3], base_shape[4]),
                tile_x0_outputs,
            )
        return blended, blended_x0

    @classmethod
    def _sample_av_window(
        cls,
        guider,
        sampler,
        sigmas,
        payload,
        latent_tile,
        latent_overlap,
        seed,
        causal_fix=False,
        progress_state: _GlobalSamplerProgress | None = None,
    ):
        video_samples = payload["video_samples"]
        audio_samples = payload["audio_samples"]
        video_noise = payload["video_noise"]
        audio_noise = payload["audio_noise"]
        video_mask = payload["video_mask"]
        audio_mask = payload["audio_mask"]
        base_video_shape = tuple(video_samples.shape)

        if causal_fix:
            print("[LTX2TiledSampler] 对非首个 AV 时间窗口应用 LTX 首帧因果补偿。", flush=True)
            video_samples = prepend_first_temporal_slice(video_samples, dim=2)
            video_noise = prepend_first_temporal_slice(video_noise, dim=2)
            video_mask = prepend_first_temporal_slice(video_mask, dim=2) if video_mask is not None else None

        height = video_samples.shape[3]
        width = video_samples.shape[4]
        regions = build_spatial_tiles(height, width, latent_tile, latent_overlap)

        video_tile_outputs = []
        video_x0_outputs = []
        audio_output = None
        audio_x0_output = None

        if len(regions) > 1:
            print("[LTX2TiledSampler] LTXAV 模式下先为当前时间窗口生成冻结音频，再对视频分支做空间切块。", flush=True)
            audio_output, audio_x0_output = cls._prime_window_audio(
                guider,
                sampler,
                sigmas,
                video_samples,
                video_noise,
                video_mask,
                audio_samples,
                audio_noise,
                audio_mask,
                regions[0],
                seed,
                progress_state=progress_state,
            )
            frozen_audio_mask = cls._make_frozen_audio_mask(audio_output)
        else:
            frozen_audio_mask = None

        for region in regions:
            tile_video_samples = slice_video_tile(video_samples, region)
            tile_video_noise = slice_video_tile(video_noise, region)
            tile_video_mask = slice_video_tile(video_mask, region) if video_mask is not None else None

            if len(regions) > 1:
                nested_samples = make_nested(tile_video_samples, audio_output)
                effective_video_mask = cls._make_video_active_mask(tile_video_samples, tile_video_mask)
                effective_audio_mask = frozen_audio_mask
            else:
                nested_samples = make_nested(tile_video_samples, audio_samples)
                effective_video_mask = tile_video_mask
                effective_audio_mask = audio_mask

            nested_noise = make_nested(tile_video_noise, audio_noise)
            nested_mask = (
                make_nested(effective_video_mask, effective_audio_mask)
                if (effective_video_mask is not None or effective_audio_mask is not None)
                else None
            )

            print(
                f"[LTX2TiledSampler] AV 时间窗口内空间 tile row={region.row} col={region.col} "
                f"v=({region.v_start}:{region.v_end}) h=({region.h_start}:{region.h_end})",
                flush=True,
            )

            sampled, x0 = cls._run_subsample(
                guider,
                sampler,
                sigmas,
                nested_samples,
                nested_noise,
                nested_mask,
                seed,
                progress_state=progress_state,
            )
            sampled_video, sampled_audio = split_av_components(sampled)
            if causal_fix:
                sampled_video = strip_first_temporal_slice(sampled_video, dim=2)
            video_tile_outputs.append((region, sampled_video))
            if audio_output is None:
                audio_output = sampled_audio
            if x0 is not None:
                x0_video, x0_audio = split_av_components(x0)
                if causal_fix:
                    x0_video = strip_first_temporal_slice(x0_video, dim=2)
                video_x0_outputs.append((region, x0_video))
                if audio_x0_output is None:
                    audio_x0_output = x0_audio

        blended_video = blend_tiles(
            (base_video_shape[0], base_video_shape[1], base_video_shape[2], base_video_shape[3], base_video_shape[4]),
            video_tile_outputs,
        )
        blended_x0 = None
        if audio_x0_output is not None and len(video_x0_outputs) == len(video_tile_outputs):
            blended_x0_video = blend_tiles(
                (base_video_shape[0], base_video_shape[1], base_video_shape[2], base_video_shape[3], base_video_shape[4]),
                video_x0_outputs,
            )
            blended_x0 = make_nested(blended_x0_video, audio_x0_output)

        return make_nested(blended_video, audio_output), blended_x0

    @classmethod
    def _concat_time(cls, parts):
        if not parts:
            raise ValueError("没有生成任何时间窗口输出。")
        first = parts[0]
        if getattr(first, "is_nested", False):
            tensors = []
            first_tensors = first.unbind()
            for index in range(len(first_tensors)):
                tensors.append(torch.cat([part.unbind()[index] for part in parts], dim=2))
            return comfy.nested_tensor.NestedTensor(tensors)
        return torch.cat(parts, dim=2)

    @classmethod
    def _sample_video(
        cls,
        guider,
        sampler,
        sigmas,
        samples,
        full_noise,
        noise_mask,
        sample_frames,
        overlap_frames,
        latent_tile,
        latent_overlap,
        seed,
        progress_state: _GlobalSamplerProgress | None = None,
    ):
        windows = build_time_windows(samples.shape[2], sample_frames, overlap_frames)
        current_samples = samples
        out_parts = []
        x0_parts = []
        has_x0 = True

        for window in windows:
            print(
                f"[LTX2TiledSampler] 处理时间窗口 index={window.index} "
                f"range=({window.history_start}:{window.end}) retain=({window.retain_start}:{window.retain_end})",
                flush=True,
            )
            window_samples, window_noise, window_mask = slice_video_window_payload(current_samples, full_noise, noise_mask, window)
            window_mask = freeze_temporal_history(window_samples, window_mask, window.retain_start, dim=2)
            if window.retain_start > 0:
                print(
                    f"[LTX2TiledSampler] 时间窗口 index={window.index} 冻结前置历史帧 {window.retain_start}，"
                    f"并复用上一窗口结果作为连续上下文。",
                    flush=True,
                )
            chunk_samples, chunk_x0 = cls._sample_video_window(
                guider,
                sampler,
                sigmas,
                window_samples,
                window_noise,
                window_mask,
                latent_tile,
                latent_overlap,
                seed,
                causal_fix=not window.is_first,
                progress_state=progress_state,
            )
            current_samples = update_video_window_context(current_samples, window, chunk_samples)
            out_parts.append(chunk_samples[:, :, window.retain_start:window.retain_end])
            if chunk_x0 is None:
                has_x0 = False
            else:
                x0_parts.append(chunk_x0[:, :, window.retain_start:window.retain_end])

        return cls._concat_time(out_parts), cls._concat_time(x0_parts) if has_x0 and x0_parts else None

    @classmethod
    def _sample_av(
        cls,
        guider,
        sampler,
        sigmas,
        samples,
        full_noise,
        noise_mask,
        sample_frames,
        overlap_frames,
        latent_tile,
        latent_overlap,
        seed,
        progress_state: _GlobalSamplerProgress | None = None,
    ):
        video_samples, _ = split_av_components(samples)
        windows = build_time_windows(video_samples.shape[2], sample_frames, overlap_frames)
        current_samples = samples
        out_video_parts = []
        out_audio_parts = []
        x0_video_parts = []
        x0_audio_parts = []
        has_x0 = True

        for window in windows:
            print(
                f"[LTX2TiledSampler] 处理 AV 时间窗口 index={window.index} "
                f"range=({window.history_start}:{window.end}) retain=({window.retain_start}:{window.retain_end})",
                flush=True,
            )
            payload = build_av_window_payload(current_samples, full_noise, noise_mask, window)
            payload["video_mask"] = freeze_temporal_history(
                payload["video_samples"],
                payload["video_mask"],
                window.retain_start,
                dim=2,
            )
            payload["audio_mask"] = freeze_temporal_history(
                payload["audio_samples"],
                payload["audio_mask"],
                payload["audio_retain_start"],
                dim=2,
            )
            if window.retain_start > 0 or payload["audio_retain_start"] > 0:
                print(
                    f"[LTX2TiledSampler] AV 时间窗口 index={window.index} 冻结视频历史 {window.retain_start} 帧，"
                    f"冻结音频历史 {payload['audio_retain_start']} 帧，并复用上一窗口结果。",
                    flush=True,
                )
            chunk_samples, chunk_x0 = cls._sample_av_window(
                guider,
                sampler,
                sigmas,
                payload,
                latent_tile,
                latent_overlap,
                seed,
                causal_fix=not window.is_first,
                progress_state=progress_state,
            )
            current_samples = update_av_window_context(current_samples, payload, chunk_samples)
            chunk_video, chunk_audio = split_av_components(chunk_samples)
            out_video_parts.append(chunk_video[:, :, window.retain_start:window.retain_end])
            out_audio_parts.append(chunk_audio[:, :, payload["audio_retain_start"]:payload["audio_retain_end"]])

            if chunk_x0 is None:
                has_x0 = False
            else:
                x0_video, x0_audio = split_av_components(chunk_x0)
                x0_video_parts.append(x0_video[:, :, window.retain_start:window.retain_end])
                x0_audio_parts.append(x0_audio[:, :, payload["audio_retain_start"]:payload["audio_retain_end"]])

        samples_out = make_nested(torch.cat(out_video_parts, dim=2), torch.cat(out_audio_parts, dim=2))
        x0_out = None
        if has_x0 and x0_video_parts and x0_audio_parts:
            x0_out = make_nested(torch.cat(x0_video_parts, dim=2), torch.cat(x0_audio_parts, dim=2))
        return samples_out, x0_out

    @classmethod
    def execute(cls, noise, guider, sampler, sigmas, latent_image, sample_frames, overlap_frames, tile_size, tile_overlap, unique_id=None) -> io.NodeOutput:
        if sample_frames < 1:
            raise ValueError("采样帧数必须 >= 1。")
        if overlap_frames < 0:
            raise ValueError("时间重叠帧数必须 >= 0。")

        latent = latent_image.copy()
        fixed_samples = comfy.sample.fix_empty_latent_channels(
            guider.model_patcher,
            latent["samples"],
            latent.get("downscale_ratio_spacial", None),
            latent.get("downscale_ratio_temporal", None),
        )
        latent["samples"] = fixed_samples
        latent_kind = detect_latent_kind(fixed_samples)
        noise_mask = latent.get("noise_mask", None)

        spatial_ratio = cls._get_spatial_ratio(latent, guider)
        latent_tile, latent_overlap = cls._convert_pixel_tiles_to_latent(tile_size, tile_overlap, spatial_ratio)

        if latent_kind == "ltxv":
            total_frames = fixed_samples.shape[2]
            latent_h = fixed_samples.shape[3]
            latent_w = fixed_samples.shape[4]
        else:
            video_samples, _ = split_av_components(fixed_samples)
            total_frames = video_samples.shape[2]
            latent_h = video_samples.shape[3]
            latent_w = video_samples.shape[4]

        use_spatial_tiles = latent_tile < max(latent_h, latent_w)
        use_time_windows = sample_frames < total_frames

        full_noise = noise.generate_noise(latent)
        sampler_steps = max(0, int(sigmas.shape[-1] - 1))
        window_count = len(build_time_windows(total_frames, sample_frames, overlap_frames))
        tile_count = len(build_spatial_tiles(latent_h, latent_w, latent_tile, latent_overlap))
        total_progress_steps = max(1, window_count * tile_count * sampler_steps)
        progress_state = _GlobalSamplerProgress(guider.model_patcher, total_progress_steps, node_id=unique_id)
        print(
            f"[LTX2TiledSampler] 全局进度模式: windows={window_count} tiles_per_window={tile_count} "
            f"sampler_steps={sampler_steps} total_progress_steps={total_progress_steps}",
            flush=True,
        )

        try:
            if not use_spatial_tiles and not use_time_windows:
                print("[LTX2TiledSampler] 命中 fast path，退化为单次完整采样。", flush=True)
                samples, x0 = cls._run_subsample(
                    guider,
                    sampler,
                    sigmas,
                    fixed_samples,
                    full_noise,
                    noise_mask,
                    noise.seed,
                    progress_state=progress_state,
                )
            else:
                if latent_kind == "ltxv":
                    samples, x0 = cls._sample_video(
                        guider,
                        sampler,
                        sigmas,
                        fixed_samples,
                        full_noise,
                        noise_mask,
                        sample_frames,
                        overlap_frames,
                        latent_tile,
                        latent_overlap,
                        noise.seed,
                        progress_state=progress_state,
                    )
                else:
                    samples, x0 = cls._sample_av(
                        guider,
                        sampler,
                        sigmas,
                        fixed_samples,
                        full_noise,
                        noise_mask,
                        sample_frames,
                        overlap_frames,
                        latent_tile,
                        latent_overlap,
                        noise.seed,
                        progress_state=progress_state,
                    )
        finally:
            progress_state.close()

        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        samples = cls._normalize_output_structure(samples, latent_kind, fixed_samples)
        out["samples"] = move_to_device(samples, comfy.model_management.intermediate_device())

        if x0 is not None:
            out_denoised = latent.copy()
            x0 = cls._normalize_output_structure(x0, latent_kind, fixed_samples)
            out_denoised["samples"] = move_to_device(x0, comfy.model_management.intermediate_device())
        else:
            out_denoised = out

        return io.NodeOutput(out, out_denoised)

    sample = execute
