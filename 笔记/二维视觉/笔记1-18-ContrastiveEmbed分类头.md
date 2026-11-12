> **归档来源**：`老笔记/笔记1.md` 第 1823–1870 行（按主题拆分归档）

---

**所属：** 模型架构 / Grounding DINO

### 18.1 它做什么

`ContrastiveEmbed` 不是传统分类头，**没有可学习参数**。它只做一件事：

```python
res = x @ y.transpose(-1, -2)   # 纯矩阵乘法，无权重
# x: [B, N_queries, 256]  ← 视觉特征（object query）
# y: [B, L, 256]          ← 文本特征（Encoder 增强后的 token embedding）
# res: [B, N_queries, L]  ← 每个 query 与每个文本 token 的点积相似度
```

然后两步后处理：
1. **masked_fill**：将 padding token 的分数置为 -∞
2. **pad to max_text_len**：补到固定长度 256

### 18.2 与 YOLO 分类头的对比

| | YOLO 分类头 | Grounding DINO ContrastiveEmbed |
|---|---|---|
| **结构** | `nn.Linear(256, 80)` 有可学习权重 W | 无参数，纯点积 `x @ y^T` |
| **类别数** | 固定 80 类（COCO），训练时就定死 | 动态，由输入文本决定 |
| **"猫"的分数来源** | `W["猫"] · x` — 权重矩阵第"猫"行的固定向量 | `x · y_cat` — 当前输入中"cat"的实时文本向量 |
| **新增类别** | 改头、重新训练 | 换个 prompt 文本就行，零样本 |
| **推理输出** | 80 个分数 | L 个分数，对应输入文本的每个 token |

### 18.3 本质区别

```
YOLO（闭集）：
  x [256] → Linear(256, 80) → [80] → argmax → "cat"
            ↑
        W 固定，80 个类别各一行，终生不变

Grounding DINO（开集）：
  x [256] · y_text^T → [L] → "cat" 是其中某个 token 的分数
  ↑         ↑
  视觉      文本 token 向量，输入变它就变
```

**一句话：** YOLO 的类别空间由 `nn.Linear` 的权重矩阵固化，Grounding DINO 的类别空间由当前输入的文本动态定义。这就是开集检测的根本——不是"从固定类里选"，而是"和当前描述文本里的词比相似度"。


---

## 19. 两阶段模式：Encoder 如何生成初始候选框
