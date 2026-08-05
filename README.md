# ComfyUI LTX2.3 Tiled Sampler Custom Advanced

一个独立的 ComfyUI 自定义节点插件，用于在 LTX 2.3 工作流中替代原生 `SamplerCustomAdvanced`，并增加时空分片采样能力。

## 功能

- 保持原始 `SamplerCustomAdvanced` 的输入输出契约
- 新增时间分片，避免一次性把全部 latent 帧送入 transformer
- 新增空间分片和重叠融合，降低高分辨率采样时的显存压力
- 支持 `LTXV` 以及 `LTXAV`

## 节点

- 节点名：`LTX23TiledSamplerCustomAdvanced`
- 显示名：`LTX2.3 Tiled Sampler Custom Advanced`

## 新增参数

- `采样帧数`
  - 默认：`32`
  - 单位：`latent 帧`
  - 含义：每个后续时间窗口新增多少 latent 帧

- `分片像素`
  - 默认：`320`
  - 单位：`像素`
  - 含义：空间分片大小
  - 内部会按 LTX 的空间缩放比例转换到 latent 空间

- `重合像素`
  - 默认：`40`
  - 单位：`像素`
  - 含义：空间分片之间的重叠宽度
  - 内部会按 LTX 的空间缩放比例转换到 latent 空间

## 隐藏规则

- 节点不会显示 `重叠帧数`
- 内部固定：
  - `重叠帧数 = 采样帧数 * 2`

因此当 `采样帧数 = 32` 时：
- 后续每个时间窗口新增 `32` 个 latent 帧
- 内部自动回看 `64` 个 latent 帧历史

## 安装

将本插件目录放入 ComfyUI 的 `custom_nodes` 下，或以软链接方式挂载到该目录，然后重启 ComfyUI。

## 与原生 SamplerCustomAdvanced 的区别

- 原生节点是一次性对整段 latent 做采样
- 本插件会：
  - 先生成完整噪声
  - 再按时间窗口切分
  - 每个时间窗口内按空间 tile 切分
  - 用线性权重融合空间重叠区域
  - 最后按时间顺序拼回完整输出

## AV 说明

在 `LTXAV` 模式下：

- 视频分支支持时间分片和空间切块
- 音频分支支持时间分片
- 当存在多个空间 tile 时，音频分支保留该时间窗口中第一个 tile 的采样结果

这是首版兼容策略，目的是先保证与默认 LTX 2.3 工作流兼容。
