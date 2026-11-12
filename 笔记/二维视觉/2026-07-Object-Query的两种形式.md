> **归档来源**：`老笔记/暂时未归纳点7月.md` 第 2237–2400 行（按主题拆分归档）

---

## 27. Object Query 的两种形式 (2026.07)

> 来源：讨论 RT-DETR v2 / Grounding DINO / DETR 时整理

### 一句话总结

> **传统 Object Query 是无空间坐标的纯向量，靠全局注意力找目标；空间 Query 是“参考点坐标 + 局部特征”，靠可变形注意力在指定位置附近 refine。**

---

### 形式一：传统 Object Query（如 DETR）

```
图像 → CNN Backbone → Transformer Encoder
                              ↓
                   300 个可学习的 object queries
                              ↓
              每个 query 和 所有 图像 token 做全局 cross-attention
                              ↓
                        输出 cls + bbox
```

**Query 是什么？**

```python
nn.Embedding(300, hidden_dim)
```

- 没有显式的 `(x, y)` 坐标
- 没有“我应该看图哪里”的先验
- 完全靠训练学会关注什么

**注意力方式：**

**全局 attention**：每个 query 和整张图的所有位置算相似度。

```
复杂度: O(num_queries × num_image_tokens)
```

如果特征图有 8000 个 token，300 个 query：

```
300 × 8000 = 2,400,000 次相似度计算
```

**优点：**

- 不受固定位置限制，可以看图里任何地方
- 对遮挡、大目标、全局上下文建模能力强

**缺点：**

- 计算量大，内存占用高
- 收敛慢（DETR 原论文要 500 epochs）
- 高分辨率特征图上速度很慢

**代表模型：** DETR

---

### 形式二：带空间坐标的 Query（如 RT-DETR v2 / Deformable DETR / Grounding DINO）

```
图像 → CNN Backbone → Hybrid Encoder 做密集预测
                              ↓
                 每个空间位置预测 class score + 粗框
                              ↓
                    选 top-K 个高分候选位置
                              ↓
           这些位置的 坐标 + 特征 作为初始 query
                              ↓
              Decoder 用可变形注意力在参考点周围采样
                              ↓
                    迭代 refine 输出最终框
```

**Query 是什么？**

```python
query = {
    "reference_point": (0.3, 0.5),  # 图像上的空间坐标
    "content_feature": [...]        # 该位置的特征向量
}
```

- 有显式的空间位置
- 从一开始就知道“我应该先看图哪里”
- 可以用可变形注意力只在附近采样

**注意力方式：**

**可变形注意力（Deformable Attention）**：每个 query 只在自己的 reference point 周围采几个点。

```
复杂度: O(num_queries × num_levels × num_points)
```

例如 300 个 query，4 个 feature level，每个点采 4 个位置：

```
300 × 4 × 4 = 4,800 次采样
```

相比全局注意力的 2,400,000，小了约 **500 倍**。

**优点：**

- 计算量小，速度快
- 收敛快（Deformable DETR 50 epochs）
- 适合高分辨率特征图
- query 有空间先验，refine 更稳定

**缺点：**

- 依赖 encoder 密集预测的质量
- 如果初始 reference point 不准，可能 refine 不回来
- 对远离初始位置的目标不敏感

**代表模型：** Deformable DETR、RT-DETR v2、Grounding DINO

---

### 两种形式的本质区别

| | 传统 Object Query | 空间 Query |
|---|---|---|
| **Query 内容** | 纯可学习向量 | 参考点坐标 + 局部特征 |
| **空间信息** | 无显式坐标 | 有显式坐标 |
| **注意力范围** | 全局所有图像 token | 参考点周围局部采样 |
| **计算复杂度** | `O(N_q × N_tokens)` | `O(N_q × K × L)` |
| **收敛速度** | 慢 | 快 |
| **空间先验** | 无 | 有 |
| **对初始质量依赖** | 低 | 高 |
| **代表模型** | DETR | Deformable DETR、RT-DETR v2、Grounding DINO |

---

### 为什么 RT-DETR v2 / Grounding DINO 选空间 Query？

1. **快**：可变形注意力避免全局 attention 的巨大计算量
2. **准**：query 从 encoder 密集预测中选出，一开始就有好的初始位置
3. **稳**：reference point 可以迭代 refine，每层都在前一层基础上微调
4. **适合高分辨率**：检测任务需要高分辨率特征图，全局 attention 吃不消

---

### 类比理解

> **传统 query 像“盲人摸象”**：没有方向，要摸遍全图才知道东西在哪。  
> **空间 query 像“已经知道大概位置，只在附近细看”**：有地图指引，效率高很多。

---

### 记忆口诀

```
DETR query 纯向量，全局 attention 慢慢学
空间 query 有坐标，可变形 attention 局部看
一个无先验全局找，一个有坐标快速 refine
实时检测用后者，精度速度两相宜
```

---
