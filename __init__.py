from comfy_api.latest import ComfyExtension

from .nodes import LTX2TiledSamplerCustomAdvanced


class LTX2TiledSamplerExtension(ComfyExtension):
    async def get_node_list(self) -> list[type]:
        return [LTX2TiledSamplerCustomAdvanced]


async def comfy_entrypoint() -> LTX2TiledSamplerExtension:
    return LTX2TiledSamplerExtension()


__all__ = ["comfy_entrypoint", "LTX2TiledSamplerExtension", "LTX2TiledSamplerCustomAdvanced"]
