> **归档来源**：`老笔记/笔记1.md` 第 1730–1781 行（按主题拆分归档）

---

**所属：** 模型架构 / Grounding DINO

### 16.1 核心区别

| | ViT（如 DETR） | Swin/ResNet（如 Grounding DINO） |
|---|---|---|
| 输出格式 | token 序列 `[B, N, C]` | 2D 特征图 `[B, C, H, W]` |
| 空间结构 | 已展平，丢失 2D 关系 | 保留，H×W 明确 |
| 位置编码 | 额外加 position embedding | 有独立的位置编码分支 `poss` |
| 投影方式 | `nn.Linear(C, d)` | `nn.Conv2d(C, d, 1)` |

即使 Swin Transformer 内部做了 self-attention，它用 window-based attention 保持了 2D 空间结构，输出**仍是特征图**。

### 16.2 Grounding DINO 完整操作链路

```
Backbone (Swin/ResNet)
    │
    ▼  多尺度 2D 特征图，各层通道数不同
features[0]: [B, 512,  100, 150]   ← Layer3
features[1]: [B, 1024, 50,  75 ]   ← Layer4
features[2]: [B, 2048, 25,  38 ]   ← 额外下采样
features[3]: [B, 2048, 13,  19 ]   ← 再下采样
    │
    ▼  input_proj：每个特征图用各自的 1×1 Conv
input_proj[0](src) → [B, 256, 100, 150]
input_proj[1](src) → [B, 256, 50,  75 ]
input_proj[2](src) → [B, 256, 25,  38 ]
input_proj[3](src) → [B, 256, 13,  19 ]
    │
    ▼  flatten：展平成 token
src.flatten(2).transpose(1,2)
[B, 256, 100, 150] → [B, 15000, 256]
[B, 256, 50,  75 ] → [B, 3750,  256]
[B, 256, 25,  38 ] → [B, 950,   256]
[B, 256, 13,  19 ] → [B, 247,   256]
    │
    ▼  concat：拼成一个大序列
total: [B, 19797, 256]
    │
    ▼  送入 Transformer Encoder
```

### 16.3 为什么 Grounding DINO 需要保留 2D 特征图？

因为 deformable attention 需要知道每个特征点的**2D 空间坐标**来计算采样位置。如果用 ViT 那种已经展平的 token，空间信息就丢失了，后续的可变形注意力无法基于 2D 参考点做稀疏采样。


---

## 17. GroupNorm：为什么 input_proj 用 GroupNorm 而不是 BatchNorm/LayerNorm
