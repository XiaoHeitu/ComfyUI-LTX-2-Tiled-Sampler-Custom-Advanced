from comfy_api.latest import ComfyExtension

from .nodes import LTX23TiledSamplerCustomAdvanced


class LTX23TiledSamplerExtension(ComfyExtension):
    async def get_node_list(self) -> list[type]:
        return [LTX23TiledSamplerCustomAdvanced]


async def comfy_entrypoint() -> LTX23TiledSamplerExtension:
    return LTX23TiledSamplerExtension()


__all__ = ["comfy_entrypoint", "LTX23TiledSamplerExtension", "LTX23TiledSamplerCustomAdvanced"]
