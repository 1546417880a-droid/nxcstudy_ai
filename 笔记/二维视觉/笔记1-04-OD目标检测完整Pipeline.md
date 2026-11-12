> **归档来源**：`老笔记/笔记1.md` 第 83–153 行（按主题拆分归档）

---

**所属：** Pipeline 流程

### 4.1 数据预处理：三种 token 类型

以模板 `"Please detect {} in this image. Answer the question with object indexes."` 为例：

| 类型 | 例子 | token ID | 从哪里获取向量 |
|---|---|---|---|
| 语言 token | `Please`, `detect`, `<region0>` | 正常词表 ID | `embed_tokens` 词嵌入矩阵查表 |
| 图像占位符 | `<image>` | `-200` (IMAGE_TOKEN_INDEX) | 主视觉塔 → mm_projector |
| 区域占位符 | `<regionfeat>` | `-300` (DEFAULT_REGION_INDEX) | 辅助塔+主塔 → HFRE → mm_projector_aux |

**训练时序列结构：**
```
[151644] [-200] [文本token...] [-300] [-300] [-300]
   ↑       ↑                     ↑     ↑     ↑
  bos   图像占位符            bbox0  bbox1  bbox2
                             特征   特征   特征
```

**推理时序列结构（多了 `<regionN>` 文本标签）：**
```
<region0><regionfeat><region1><regionfeat><region2><regionfeat>
```

### 4.2 `<regionN>` 的双重作用

`<region0>` 是一个真实 token，会经过 `embed_tokens` 变成词嵌入向量。模型通过两重信息定位"这是第几个候选区域"：

- **隐式**：`-300` 在序列中的出现顺序 = `bbox_list` 的遍历顺序（严格按序对应）
- **显式**：`<region0>`, `<region1>` 文本 token 经 embed_tokens 变成嵌入向量，让模型直接读出索引号

### 4.3 区域特征提取：模型不自己找位置

OD 流程是两阶段的：

1. **外部检测器（SAM3/UPN）生成候选框** → 模型只做"验证"，判断哪些候选框包含目标物体
2. **HFRE 用 ROIAlign 提取每个框的特征**：

```
DaViT 4层特征                      Qwen ViT 4层特征
    │                                    │
    ▼ concat                             ▼ concat
  [1, 3840, H, W]                     [1, 5120, H, W]
    │                                    │
    ▼ ROIAlign(box, 0.25, 7)            ▼ ROIAlign(vt_boxes, 1/14, 7)
  [N, 3840, 7, 7]                     [N, 5120, 7, 7]
    │                                    │
    ▼ mean(dim=(2,3))                    ▼ mean(dim=(2,3))
  [N, 3840]                           [N, 5120]
    │                                    │
    └────────── concat ──────────────────┘
                    │
                    ▼  nn.Linear → [N, 4096]
                    ▼  替换 input_ids 中的 -300
```

### 4.4 bbox 坐标变换链

| 阶段 | 坐标空间 | 函数 |
|---|---|---|
| 原始数据 | 像素坐标 [x1,y1,x2,y2] | COCO 格式 |
| 图像 resize | 按比例缩放 | `resize_shortest_edge_images_and_bboxes` |
| 辅助塔预处理 | 768×768 | `adjust_bbox` |
| 主塔坐标再缩放 | 动态分辨率 | `encode_regions` 中 scale_height/scale_width |
| HFRE 内部 | 归一化到 [0,1]，转中心点+宽高 | ROIAlign 前 |

---

## 5. 位置编码系统
