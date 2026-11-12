> **归档来源**：`老笔记/暂时未归纳点7月.md` 第 263–416 行（按主题拆分归档）

---

## 9. LocateAnything 前处理与 ViT 架构详解

> 来源：逐行分析 `locateanything_worker.py` + `modeling_vit.py` + `processing_locateanything.py`

### 9.1 推理入口的配置参数

```python
LocateAnythingWorker(
    model_path="nvidia/LocateAnything-3B",
    attn="la_flash",          # LLM attention 实现 (FlashAttention varlen)
    vision_attn="auto",       # Vision encoder attention (自动选择后端)
    scheduler="pipeline",     # batch 调度策略 (batch_utils 中定义)
)
```

| 参数 | 作用域 | 含义 |
|---|---|---|
| `attn` | LLM（语言模型） | 控制文本/多模态的 attention 后端，`la_flash`=FlashAttention varlen |
| `vision_attn` | Vision Encoder | 控制 ViT 内部 attention 后端，`auto`=自动选择 |
| `scheduler` | batch 推理 | 混合 MTP/NTP 的调度策略，当前仅暴露 `pipeline` |

两者独立配置，对应环境变量 `LA_FLASH_ATTN` 和 `LA_FLASH_VISION_ATTN`。

### 9.2 任务方法选择机制

8 个任务便捷方法（`detect`, `ground_single`, `ground_multi`, `ground_text`, `detect_text`, `ground_gui`, `point`, `detect_visual_prompt`）**没有自动路由**，方法选择完全由调用方决定。

核心机制是 **prompt 驱动**——所有方法共享同一个 `predict()`，区别仅在于拼进去的指令文本：

```
detect()       → "Locate all the instances that matches the following description: {cats}."
ground_single()→ "Locate a single instance that matches the following description: {phrase}."
ground_multi() → "Locate all the instances that match the following description: {phrase}."
ground_text()  → "Please locate the text referred as {phrase}."
detect_text()  → "Detect all the text in box format."
ground_gui()   → "Point to: {phrase}." (point) / "Locate the region..." (box)
point()        → "Point to: {phrase}."
```

`ground_gui(output_type="point")` vs `ground_gui(output_type="box")` 的切换是纯 prompt 级别——不同指令 → 模型输出不同格式（`<box><x><y></box>` vs `<box><x1><y1><x2><y2></box>`），后处理用不同正则解析。

### 9.3 完整前处理管线

**图像侧（3 步）**：
1. `_build_messages()` → 将 PIL Image 放入 `{"type": "image", "image": image}`，成为 `<image-1>`
2. `process_vision_info()` → 提取图像，统一转 PIL
3. `processor(images=images)` → `rescale → normalize → patchify → pixel_values + image_grid_hws`

**patchify 输出**: `[H*W, C, patch_size, patch_size]`。336×336 图 + patch_size=14 → `[576, 3, 14, 14]`。每个 patch 是 3×14×14 的小方块，**不是一个像素**。

**多图 concat**: 所有图的 patch 在 dim=0 上堆叠。3 张图各 576 个 patch → `[1728, 3, 14, 14]`。

**文本侧（3 步）**：
1. `_build_messages()` → `{"type": "text", "text": question}`
2. `py_apply_chat_template()` → 渲染为 ChatML 格式（`<|im_start|>user\n<image-1>\n{question}<|im_end|>`）
3. `processor(text=[text])`:
   - `replace_media_placeholder`: `<image-1>` → `<img><IMG_CONTEXT>×N</img>`（N = H*W/merge_kernel_size²）
   - `tokenizer` → input_ids + attention_mask

最终给模型的：
```python
{
    "input_ids":      [..., IMG_CONTEXT(xN), ...],     # 文本 token + 图像占位符
    "pixel_values":   [P_total, 3, 14, 14],            # 原始 patch 像素
    "image_grid_hws": [(24,24), (24,24), (24,24)],     # 每张图的 patch grid 尺寸
}
```

### 9.4 IMG_CONTEXT 的两阶段替换

```
阶段1 (tokenizer): <image-1> → N 个 IMG_CONTEXT token（同一个 ID，仅占位）
阶段2 (forward):   IMG_CONTEXT 的 embedding 被 ViT 产出的视觉特征硬替换
```

核心代码 `modeling_locateanything.py:205-242`：

```python
input_embeds = self.language_model.get_input_embeddings()(input_ids)  # token→embed
vit_embeds = self.extract_feature(pixel_values, image_grid_hws)        # ViT 前向
vit_embeds = self.mlp1(vit_embeds)                                     # MLP 投影
selected = (input_ids == self.image_token_index)                       # 找到 IMG_CONTEXT
input_embeds[selected] = vit_embeds[:selected.sum()]                   # 硬替换
outputs = self.language_model(inputs_embeds=input_embeds, ...)         # 送入 LLM
```

本质：tokenizer 挖坑 → ViT 产内容 → forward 时填坑。

### 9.5 为什么需要 image_grid_hws

`pixel_values` 是**无结构的扁平数据**（所有 patch 压在 dim=0），`image_grid_hws` 是恢复结构的钥匙：

1. **拆分多图**：`patch_merger` 按 H×W 切出每张图的 patch，reshape 成 2D 做合并
2. **2D RoPE 位置编码**：需要知道每个 patch 的 (row, col)，只知序号无法推算
3. **确定 IMG_CONTEXT 数量**：N = H×W / merge_kernel_size²

### 9.6 ViT 如何处理 patch

**ViT 不会再重新切图**——image_processor 已经切好，ViT 直接用。

- **patch_embed**: `Conv2d(3, D, kernel=14, stride=14)` 对每个 `[3,14,14]` 小块独立卷积 → `[1728, D]`，然后加 2D 位置编码

- **Transformer Encoder**: 一维序列 + **块对角注意力 mask** 隔离多图：
```
        图1(576)  图2(576)  图3(576)
图1       ████      ░░░░      ░░░░       ← 图1 的 patch 只互看
图2       ░░░░      ████      ░░░░       ← 图2 同理
图3       ░░░░      ░░░░      ████       ← 图3 同理
```
packed sequence + block-diagonal mask，无 padding 浪费，mask 保证独立。

- **patch_merger**: 用 grid_hws 恢复 2D → 2×2 merge → 如 24×24=576 → 12×12=144 个向量/图

最终 ViT 输出：`List[[144, D], [144, D], [144, D]]`，各图独立。

### 9.7 位置编码对比

| | LocateAnything | Qwen2-VL |
|---|---|---|
| ViT 内 | 2D RoPE（row+col） | 2D RoPE |
| LLM 侧 | 复用 1D RoPE | 带 2D row/col 位置信息 |
| LLM 感知空间 | 间接（靠 ViT embedding 携带） | 直接（显式 visual position） |

### 9.8 全链路数据流

```
PIL Image                     Text
    │                           │
image_processor           chat_template → ChatML
  patchify                <image-1> 占位
    │                           │
pixel_values               replace_media_placeholder
[P_total,3,14,14]          <image-1> → IMG_CONTEXT×N
+ image_grid_hws                │
    │                       tokenizer → input_ids
    └──────────┬────────────────┘
               ↓
          model.forward()
    ┌──────────┴──────────┐
    │ ViT:                 │  LLM:
    │  patch_embed         │  token embeddings
    │  transformer         │  IMG_CONTEXT ← vit_embeds
    │  (2D RoPE + mask)    │  → transformer layers
    │  patch_merger        │  → PBD/MTP 解码
    │  → MLP1 → vit_embeds │
    └──────────────────────┘
               ↓
         解码输出文本
    <box><100><200><300><400></box>
               ↓
    parse_boxes() → 像素坐标
```

---
