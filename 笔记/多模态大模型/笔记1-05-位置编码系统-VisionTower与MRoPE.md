> **归档来源**：`老笔记/笔记1.md` 第 154–240 行（按主题拆分归档）

---

**所属：** 模型架构 / 位置编码

**描述：** 模型中有两套**完全独立**的位置编码系统，服务于不同模块。

### 5.1 Vision Tower（图像编码器）—— 2D RoPE

**核心函数**：`rot_pos_emb`（modeling_qwen2_5_vl.py:674-718）

- 为每个 image patch 生成 (h, w) 二维坐标
- (h, w) 作为索引查一张共享的 1D 频率表
- 高度和宽度方向共享同一套频率基底，仅索引值不同
- `spatial_merge_size=2` 时做 reshape+permute，让将被 merge 的 2x2 patch 获得相邻位置
- **实现类**：`Qwen2_5_VisionRotaryEmbedding`
- **应用函数**：`apply_rotary_pos_emb_vision`

```
一张 20x30 个 patch 的图:
  patch(i,j) 的 Q,K 被施加:
    cos(h_i, w_j) * Q  +  sin(h_i, w_j) * rotate_half(Q)
这样 ViT attention 就知道 patch 之间的 2D 空间关系。
```

### 5.2 LLM 解码器 —— 3D MRoPE（Multimodal RoPE）

**核心函数**：`get_rope_index`（modeling_qwen2_5_vl.py:1906-2045）

`position_ids` 形状为 `[3, batch, seq_len]`：

| 通道 | 文本 token | 图像 patch token |
|---|---|---|
| 通道0 (T) | = 线性位置 | = 0（单帧图像） |
| 通道1 (H) | = 线性位置 | = row index |
| 通道2 (W) | = 线性位置 | = col index |

文本 token 三个通道值完全相同 → 退化为普通 1D RoPE。

**`mrope_section` 切分 head_dim：**

```
mrope_section = [24, 20, 20] x2 = [48, 40, 40]
head_dim(128) = [0..47]时间 + [48..87]高度 + [88..127]宽度
cos/sin 最终排列: [T0, H0, W0, T1, H1, W1, ...]
```

**为什么文本三个通道值相同？**
因为 `mrope_section` 是固定的——无论 token 是文本还是图像，head_dim 都按同一比例切分给 T/H/W。文本没有真正的 2D 位置，三通道填相同值，效果上等价于普通 1D RoPE，实现上避免了对文本和图像分别走不同路径。

- **实现类**：`Qwen2_5_VLRotaryEmbedding`
- **应用函数**：`apply_multimodal_rotary_pos_emb`

### 5.3 两套位置编码对比

| | Vision Tower | LLM Decoder |
|---|---|---|
| 类型 | 2D RoPE | 3D MRoPE |
| 位置维度 | (h, w) 共享频率表 | (T, H, W) 三通道独立频率 |
| head_dim 切分 | 不切分 | 按 `mrope_section` 切三段 |
| 文本处理 | N/A | 三通道填相同值 |
| 实现类 | `Qwen2_5_VisionRotaryEmbedding` | `Qwen2_5_VLRotaryEmbedding` |
| 位置 ID 来源 | `rot_pos_emb(grid_thw)` | `get_rope_index(input_ids, image_grid_thw, ...)` |

**关键不干扰机制：** Vision Tower 的 2D RoPE 在 ViT 内部就已消费完毕，输出给 LLM 的是不含位置信息的纯语义特征向量。LLM 拿到后再用 3D MRoPE 给整个多模态序列（文本+图像+区域）统一编码位置。

### 5.4 `rope_deltas` 机制

由于图像的 position_ids（含 2D 空间坐标）可能与序列长度不一致，在 autoregressive generation 的首步（prefill）计算 `rope_deltas = max(position_ids) - seq_len`，后续 steps 用它偏移新生成 token 的位置 ID。

---

## 六、关键文件索引

| 文件 | 作用 |
|---|---|
| `vlm_fo1/model/language_model/omchat_qwen2_5_vl.py` | OmChat CausalLM 主类：forward, encode_images/regions, prepare_inputs |
| `vlm_fo1/model/omchat_arch.py` | OmChatMetaModel（backbone 基类），OmChatMetaForCausalLM（CausalLM facade） |
| `vlm_fo1/model/multimodal_encoder/qwen2_5_vl/modeling_qwen2_5_vl.py` | ViT, LLM decoder, RoPE 全套, get_rope_index |
| `vlm_fo1/model/multimodal_encoder/qwen2_5_vl_encoder.py` | Qwen2_5_VlVisionTower 封装 |
| `vlm_fo1/model/multimodal_visual_prompt_encoder/hybrid_finegrained_region_encoder.py` | HFRE 混合细粒度区域编码器 |
| `vlm_fo1/task_templates.py` | OD/REC/Grounding 等任务模板 |
| `vlm_fo1/mm_utils.py` | 推理预处理：prepare_inputs, make_message_context, tokenize |
| `vlm_fo1/constants.py` | 特殊 token 常量定义 |
| `data/dataset.py` | 训练数据加载与预处理 |

---

## 7. 视觉 Token 位置信息的两种传递方式
