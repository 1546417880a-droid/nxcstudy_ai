> **归档来源**：`老笔记/暂时未归纳点7月.md` 第 418–474 行（按主题拆分归档）

---

## 10. image_grid_hws 在 MoonViT 中的三个作用

> 来源：分析 `modeling_vit.py` 中 `image_grid_hws` 的每一次使用

`image_grid_hws` 是 shape `(N, 2)` 的 tensor，记录每张图的 patch 网格尺寸 `(h, w)`，在多图 packed 场景中充当序列分隔符。

### 作用①：可学习位置编码插值（Learnable2DInterpPosEmb）

```python
# Learnable2DInterpPosEmb.forward(x, grid_hws)
for shape in grid_hws.tolist():        # e.g. [(40,60), (20,20)]
    if shape == (64, 64):
        pos_embs.append(weight.flatten(start_dim=1))   # 原始尺寸，直接用
    else:
        pos_embs.append(F.interpolate(                  # 双三次插值缩放
            weight.permute(2,0,1).unsqueeze(0),
            size=shape, mode='bicubic'
        ).squeeze(0).permute(1,2,0).flatten())
```

不同分辨率的图各自从 64×64 的权重网格上插值出对应尺寸的 2D 位置编码。训练时 64×64 直接取用，推理时其他尺寸通过双三次插值适配。

### 作用②：RoPE 频率 + cu_seqlens（MoonVitEncoder）

```python
# Step A: 2D RoPE — 每张图取各自区域的 freqs_cis
freqs_cis = self.rope_2d.get_freqs_cis(grid_hws)
# 内部: for (h,w) in grid_hws: 取 freqs_cis[:h, :w] → flatten → cat
# 结果: (Σh*w, head_dim//2)，每个 patch 有独立的 (x,y) 旋转频率

# Step B: cu_seqlens — 累积序列长度
lengths = torch.cat([torch.zeros(1), grid_hws[:,0] * grid_hws[:,1]])
cu_seqlens = lengths.cumsum(dim=0, dtype=torch.int32)
# e.g. [0, 1600, 2000] 表示图1占 [0,1600)，图2占 [1600,2000)
```

flash_attn 用 cu_seqlens 实现 block-diagonal attention：同图内部互相 attend（causal=False），不同图之间隔离。用 3 个整数替代 N×N 的 attention mask 矩阵。

### 作用③：patch_merger 拆分（patch_merger 函数）

```python
pre_sum = 0
for (h, w) in grid_hws.tolist():
    seq = x[pre_sum : pre_sum + h*w]       # 按图的边界切分
    # reshape → permute → merge 2×2 → 通道×4
    # (h, w, D) → (h/2, w/2, D×4) → (h/2 * w/2, D×4)
    outputs.append(merged)
    pre_sum += h * w
```

把 packed 序列按图的边界拆开，各自做 patch merge，最终返回 `List[Tensor]`。

### 一句话总结

没有 `image_grid_hws`，`pixel_values` 就是一堆不知道谁属于谁的像素块——位置编码不知道往哪插，attention 不知道哪些能互看，merger 不知道在哪切分。

---
