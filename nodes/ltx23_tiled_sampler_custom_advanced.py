from __future__ import annotations

from comfy_api.latest import io
import torch
import comfy.model_management
import comfy.sample
import comfy.utils
import comfy.nested_tensor
import latent_preview
import nodes as comfy_nodes

from ..utils.latent_ops import (
    build_av_window_payload,
    build_time_windows,
    detect_latent_kind,
    make_nested,
    move_to_device,
    slice_video_tile,
    slice_video_window_payload,
    split_av_components,
)
from ..utils.tile_blending import blend_tiles, build_spatial_tiles


class LTX23TiledSamplerCustomAdvanced(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="LTX23TiledSamplerCustomAdvanced",
            display_name="LTX2.3 Tiled Sampler Custom Advanced",
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
                    tooltip="每个后续时间窗口新增的 latent 帧数。内部会自动使用 2 倍该值作为隐藏重叠帧数。",
                ),
                io.Int.Input(
                    "tile_size",
                    default=320,
                    min=32,
                    max=comfy_nodes.MAX_RESOLUTION,
                    step=8,
                    tooltip="空间分片尺寸，使用像素单位；内部会按 LTX latent 缩放比例转换为 latent 空间 tile。",
                ),
                io.Int.Input(
                    "tile_overlap",
                    default=40,
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
        x0_out = guider.model_patcher.model.process_latent_out(raw_x0.cpu())
        if getattr(sampled, "is_nested", False):
            latent_shapes = [x.shape for x in sampled.unbind()]
            x0_out = comfy.nested_tensor.NestedTensor(comfy.utils.unpack_latents(x0_out, latent_shapes))
        return move_to_device(x0_out, comfy.model_management.intermediate_device())

    @classmethod
    def _run_subsample(cls, guider, sampler, sigmas, latent_samples, sub_noise, sub_mask, seed):
        x0_output = {}
        callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
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
    def _sample_video_window(cls, guider, sampler, sigmas, window_samples, window_noise, window_mask, latent_tile, latent_overlap, seed):
        height = window_samples.shape[3]
        width = window_samples.shape[4]
        regions = build_spatial_tiles(height, width, latent_tile, latent_overlap)

        tile_outputs = []
        tile_x0_outputs = []

        for region in regions:
            print(
                f"[LTX23TiledSampler] 时间窗口内空间 tile row={region.row} col={region.col} "
                f"v=({region.v_start}:{region.v_end}) h=({region.h_start}:{region.h_end})",
                flush=True,
            )
            tile_samples = slice_video_tile(window_samples, region)
            tile_noise = slice_video_tile(window_noise, region)
            tile_mask = slice_video_tile(window_mask, region) if window_mask is not None else None
            sampled, x0 = cls._run_subsample(guider, sampler, sigmas, tile_samples, tile_noise, tile_mask, seed)
            tile_outputs.append((region, sampled))
            if x0 is not None:
                tile_x0_outputs.append((region, x0))

        blended = blend_tiles(
            (window_samples.shape[0], window_samples.shape[1], window_samples.shape[2], height, width),
            tile_outputs,
        )
        blended_x0 = None
        if tile_x0_outputs and len(tile_x0_outputs) == len(tile_outputs):
            blended_x0 = blend_tiles(
                (window_samples.shape[0], window_samples.shape[1], window_samples.shape[2], height, width),
                tile_x0_outputs,
            )
        return blended, blended_x0

    @classmethod
    def _sample_av_window(cls, guider, sampler, sigmas, payload, latent_tile, latent_overlap, seed):
        video_samples = payload["video_samples"]
        audio_samples = payload["audio_samples"]
        video_noise = payload["video_noise"]
        audio_noise = payload["audio_noise"]
        video_mask = payload["video_mask"]
        audio_mask = payload["audio_mask"]

        height = video_samples.shape[3]
        width = video_samples.shape[4]
        regions = build_spatial_tiles(height, width, latent_tile, latent_overlap)

        video_tile_outputs = []
        video_x0_outputs = []
        audio_output = None
        audio_x0_output = None

        if len(regions) > 1:
            print("[LTX23TiledSampler] LTXAV 模式下仅对视频分支做空间切块，音频保留首个 tile 的结果。", flush=True)

        for region in regions:
            tile_video_samples = slice_video_tile(video_samples, region)
            tile_video_noise = slice_video_tile(video_noise, region)
            tile_video_mask = slice_video_tile(video_mask, region) if video_mask is not None else None
            nested_samples = make_nested(tile_video_samples, audio_samples)
            nested_noise = make_nested(tile_video_noise, audio_noise)
            nested_mask = make_nested(tile_video_mask, audio_mask) if (tile_video_mask is not None or audio_mask is not None) else None

            print(
                f"[LTX23TiledSampler] AV 时间窗口内空间 tile row={region.row} col={region.col} "
                f"v=({region.v_start}:{region.v_end}) h=({region.h_start}:{region.h_end})",
                flush=True,
            )

            sampled, x0 = cls._run_subsample(guider, sampler, sigmas, nested_samples, nested_noise, nested_mask, seed)
            sampled_video, sampled_audio = split_av_components(sampled)
            video_tile_outputs.append((region, sampled_video))
            if x0 is not None:
                x0_video, x0_audio = split_av_components(x0)
                video_x0_outputs.append((region, x0_video))
                if audio_x0_output is None:
                    audio_x0_output = x0_audio
            if audio_output is None:
                audio_output = sampled_audio

        blended_video = blend_tiles(
            (video_samples.shape[0], video_samples.shape[1], video_samples.shape[2], height, width),
            video_tile_outputs,
        )
        blended_x0 = None
        if audio_x0_output is not None and len(video_x0_outputs) == len(video_tile_outputs):
            blended_x0_video = blend_tiles(
                (video_samples.shape[0], video_samples.shape[1], video_samples.shape[2], height, width),
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
    def _sample_video(cls, guider, sampler, sigmas, samples, full_noise, noise_mask, sample_frames, latent_tile, latent_overlap, seed):
        windows = build_time_windows(samples.shape[2], sample_frames, sample_frames * 2)
        out_parts = []
        x0_parts = []
        has_x0 = True

        for window in windows:
            print(
                f"[LTX23TiledSampler] 处理时间窗口 index={window.index} "
                f"range=({window.history_start}:{window.end}) retain=({window.retain_start}:{window.retain_end})",
                flush=True,
            )
            window_samples, window_noise, window_mask = slice_video_window_payload(samples, full_noise, noise_mask, window)
            chunk_samples, chunk_x0 = cls._sample_video_window(
                guider, sampler, sigmas, window_samples, window_noise, window_mask, latent_tile, latent_overlap, seed
            )
            out_parts.append(chunk_samples[:, :, window.retain_start:window.retain_end])
            if chunk_x0 is None:
                has_x0 = False
            else:
                x0_parts.append(chunk_x0[:, :, window.retain_start:window.retain_end])

        return cls._concat_time(out_parts), cls._concat_time(x0_parts) if has_x0 and x0_parts else None

    @classmethod
    def _sample_av(cls, guider, sampler, sigmas, samples, full_noise, noise_mask, sample_frames, latent_tile, latent_overlap, seed):
        video_samples, _ = split_av_components(samples)
        windows = build_time_windows(video_samples.shape[2], sample_frames, sample_frames * 2)
        out_video_parts = []
        out_audio_parts = []
        x0_video_parts = []
        x0_audio_parts = []
        has_x0 = True

        for window in windows:
            print(
                f"[LTX23TiledSampler] 处理 AV 时间窗口 index={window.index} "
                f"range=({window.history_start}:{window.end}) retain=({window.retain_start}:{window.retain_end})",
                flush=True,
            )
            payload = build_av_window_payload(samples, full_noise, noise_mask, window)
            chunk_samples, chunk_x0 = cls._sample_av_window(
                guider, sampler, sigmas, payload, latent_tile, latent_overlap, seed
            )
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
    def execute(cls, noise, guider, sampler, sigmas, latent_image, sample_frames, tile_size, tile_overlap) -> io.NodeOutput:
        if sample_frames < 1:
            raise ValueError("采样帧数必须 >= 1。")

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

        if not use_spatial_tiles and not use_time_windows:
            print("[LTX23TiledSampler] 命中 fast path，退化为单次完整采样。", flush=True)
            samples, x0 = cls._run_subsample(guider, sampler, sigmas, fixed_samples, full_noise, noise_mask, noise.seed)
        else:
            if latent_kind == "ltxv":
                samples, x0 = cls._sample_video(
                    guider, sampler, sigmas, fixed_samples, full_noise, noise_mask, sample_frames, latent_tile, latent_overlap, noise.seed
                )
            else:
                samples, x0 = cls._sample_av(
                    guider, sampler, sigmas, fixed_samples, full_noise, noise_mask, sample_frames, latent_tile, latent_overlap, noise.seed
                )

        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = move_to_device(samples, comfy.model_management.intermediate_device())

        if x0 is not None:
            out_denoised = latent.copy()
            out_denoised["samples"] = move_to_device(x0, comfy.model_management.intermediate_device())
        else:
            out_denoised = out

        return io.NodeOutput(out, out_denoised)

    sample = execute
