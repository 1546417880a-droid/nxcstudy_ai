# FCOS3D 深度学习笔记

## 1. 模型概述

FCOS3D 是一种**单目相机 3D 目标检测**模型（论文: [FCOS3D](https://arxiv.org/abs/2104.10956)），将 2D anchor-free 检测器 FCOS 扩展到 3D 空间。输入单张相机图像，输出图像中物体在相机坐标系下的 3D 边框。

- **支持数据集**: nuScenes（10类）、KITTI
- **预训练权重**: 有，位于 `configs/fcos3d/metafile.yml`
- **定位**: nuScenes 3D 检测榜单 Camera-only 赛道的基础模型，PGD 等方法直接继承其架构

---

## 2. 核心设计哲学：将相机内参与网络学习解耦

FCOS3D 最核心的设计是**让神经网络不直接感知相机内参 `cam2img`**，而是通过前/后处理来桥接 2D 图像和 3D 空间。这种"编解码"范式带来两个好处：

1. 网络学习的是**与相机无关的几何表示**，泛化能力更强
2. 同一个模型可以处理不同相机参数的数据，无需重新训练

具体来说：
- **网络本身**：纯 CNN，输入只有图像像素，输出是 2.5D 编码（像素坐标 + 深度 + 尺寸 + 朝向等）
- **相机内参 `cam2img`**：在前处理中将 3D GT **投影**为 2.5D 编码，在后处理中将预测的 2.5D 编码**反投影**回 3D 空间

---

## 3. 模型结构

```
输入图像 (N, 3, H, W)
      │
      ▼
┌─────────────────────────────────────┐
│  Backbone: ResNet101 + DCNv2        │
│  - 4个 stage，输出 C2~C5            │
│  - stage3/4 使用可变形卷积 DCNv2    │
│  - strides: [4, 8, 16, 32]         │
│  - 输出通道: [256, 512, 1024, 2048] │
└─────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────┐
│  Neck: FPN (Feature Pyramid Network)│
│  - 输入: C2~C5                      │
│  - 输出: P3~P7 共 5 层              │
│  - channels: 256                    │
│  - strides: [8, 16, 32, 64, 128]   │
└─────────────────────────────────────┘
      │ (5 个多尺度特征图)
      ▼
┌─────────────────────────────────────┐
│  Head: FCOSMono3DHead               │
│  (anchor-free, per-pixel 预测)      │
│                                      │
│  共享卷积塔 (stacked_convs=2):       │
│  ├─ cls_convs (分类塔)              │
│  └─ reg_convs (回归塔)              │
│                                      │
│  分类预测: cls_score (C=10)          │
│                                      │
│  回归预测 (5个分支, 各由独立子塔处理): │
│  ├─ offset (2): Δx, Δy 像素偏移     │
│  ├─ depth  (1): 深度 z              │
│  ├─ size   (3): 3D尺寸 w, l, h      │
│  ├─ rot    (1): 朝向角 sin(θ)       │
│  └─ velo   (2): 速度 vx, vy         │
│  → concat → bbox_pred (9)           │
│                                      │
│  辅助预测:                           │
│  ├─ centerness (1): 中心度           │
│  ├─ dir_cls    (2): 方向二分类       │
│  └─ attr_pred  (9): 属性分类        │
└─────────────────────────────────────┘
```

### 代码对应关系

| 组件 | 文件 | 类名 |
|------|------|------|
| 检测器 | `mmdet3d/models/detectors/fcos_mono3d.py` | `FCOSMono3D` |
| 父类检测器 | `mmdet3d/models/detectors/single_stage_mono3d.py` | `SingleStageMono3DDetector` |
| Head | `mmdet3d/models/dense_heads/fcos_mono3d_head.py` | `FCOSMono3DHead` |
| 父类 Head | `mmdet3d/models/dense_heads/anchor_free_mono3d_head.py` | `AnchorFreeMono3DHead` |
| BBox Coder | `mmdet3d/models/task_modules/coders/fcos3d_bbox_coder.py` | `FCOS3DBBoxCoder` |
| 数据预处理器 | `mmdet3d/models/data_preprocessors/data_preprocessor.py` | `Det3DDataPreprocessor` |
| 数据打包 | `mmdet3d/datasets/transforms/formating.py` | `Pack3DDetInputs` |

### 9 维回归目标

`group_reg_dims = (2, 1, 3, 1, 2)` — 对应 5 个回归分组：

| 维度索引 | 分组 | 名称 | 含义 |
|----------|------|------|------|
| 0-1 | offset | Δx, Δy | 像素相对于 GT 2D 投影中心的偏移量 |
| 2 | depth | z | 深度（对数空间，exp 解码） |
| 3-5 | size | w, l, h | 3D 框长宽高（对数空间，exp 解码） |
| 6 | rot | sin(θ) | 朝向角 sin 编码 |
| 7-8 | velo | vx, vy | 速度（nuScenes 特有） |

### Config 文件

| 文件 | 内容 |
|------|------|
| `configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d.py` | nuScenes 训练配置 |
| `configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d_finetune.py` | nuScenes 微调配置 |
| `configs/_base_/models/fcos3d.py` | 模型基础配置 |

---

## 4. 训练流程：编码 → 前向 → 损失

训练时，3D GT 先通过相机内参**编码**为网络可学习的 2.5D 表示，网络在编码空间直接与之对比计算损失，无需在线解码回 3D。

### 4.1 前处理 — GT 的"编码" (使用 cam2img 投影)

**数据 pipeline**（`fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d.py:19-37`）：

```
nuScenes 样本 (图像 + LiDAR标注 + 相机标定)
      │
      ▼
┌──────────────────────────────────────────┐
│  LoadImageFromFileMono3D                 │
│  加载单张相机图像 → img (H, W, 3)        │
│  提取 cam2img, lidar2cam 等相机参数到meta │
└──────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────┐
│  LoadAnnotations3D                       │
│  加载 GT 标注:                           │
│  ├─ gt_bboxes:     (M, 4)  2D bboxes   │
│  ├─ gt_bboxes_3d:  (M, 9)  3D bboxes   │ ← [x,y,z,w,l,h,yaw,vx,vy]
│  ├─ gt_labels_3d:  (M,)    3D 类别标签  │
│  ├─ attr_labels:   (M,)    属性标签     │
│  └─ ✅ 用 cam2img 投影 3D中心→2D图像:   │
│      centers_2d:   (M, 2)  投影坐标(u,v)│ ← 关键编码!
│      depths:       (M,)    深度值 z     │ ← 关键编码!
└──────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────┐
│  Resize (1600×900) + RandomFlip3D        │
│  图像增强，同步更新 bboxes & centers_2d   │
└──────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────┐
│  Det3DDataPreprocessor                   │
│  BGR→RGB, Normalize, Pad → (N,3,H,W)    │
└──────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────────┐
│  Pack3DDetInputs → 打包为:              │
│  inputs = {'imgs': Tensor(N,3,H,W)}      │
│  data_samples.gt_instances_3d:           │
│    bboxes_3d, labels_3d,                │
│    centers_2d, depths, attr_labels      │
│  data_samples.metainfo: cam2img, ...    │
└──────────────────────────────────────────┘
```

**关键点**：`centers_2d` 和 `depths` 是由 `LoadAnnotations3D` 在数据加载阶段，通过 `cam2img` 将 3D GT 中心 `(x, y, z)` 投影到图像平面得到的。这一步之后，GT 就被表示成了网络可以直接回归的 2.5D 形式。

### 4.2 模型前向 — 纯视觉感知（不使用 cam2img）

**代码**: `FCOSMono3D.extract_feat()` + `FCOSMono3DHead.forward()` 

```
输入: imgs (N, 3, H, W)   ← 只有图像像素

Backbone (ResNet101+DCNv2) → C2~C5
Neck (FPN) → P3~P7, 5层, channels=256, strides=[8,16,32,64,128]

Head: 对每层 FPN 特征图并行执行 forward_single()
  ├─ cls_score   (N, 10, Hi, Wi)  — 类别分类
  ├─ bbox_pred   (N, 9,  Hi, Wi)  — 9维回归 (与GT编码格式一致)
  ├─ dir_cls     (N, 2,  Hi, Wi)  — 方向二分类
  ├─ attr_pred   (N, 9,  Hi, Wi)  — 属性分类
  └─ centerness  (N, 1,  Hi, Wi)  — 中心度

输出: 5 组预测，每组对应一个 FPN 层，格式与 GT 编码完全对齐
```

**完全不使用 cam2img**。网络的输入只有图像像素，它学习的是从像素到 2.5D 编码的映射。

### 4.3 损失计算 — 编码空间内的直接对比

**代码**: `FCOSMono3DHead.loss_by_feat()` (`fcos_mono3d_head.py:269-482`)

训练时 Loss 在**编码空间**直接计算，无需将预测解码回 3D，也无需 `cam2img` 参与：

```
网络预测(2.5D编码) ←→ GT(2.5D编码，由前处理"投影"得到)
         │
         ▼ 直接在编码空间对比
      loss_dict
```

**正负样本分配** (`_get_target_single()`, `fcos_mono3d_head.py:845-958`)：

```
每个 FPN 层的每个像素点 → 匹配最佳 GT

条件:
  ① Center Sampling: 像素必须在 GT center_2d 周围 (半径 = 1.5 × stride)
  ② Regress Range: 当前 FPN 层负责特定尺度范围
     stride=8:   [0, 48)      ← 小物体
     stride=16:  [48, 96)
     stride=32:  [96, 192)
     stride=64:  [192, 384)
     stride=128: [384, INF)   ← 大物体
  ③ 取距离最近的 GT (处理一个像素匹配多个 GT 的情况)

满足条件 → 正样本 (计算回归 loss)
不满足   → 负样本/背景 (仅计算分类 loss)
```

**构建回归目标** (`_get_target_single()`):

```python
# 对每个正样本像素 (x_grid, y_grid), 匹配到 GT 后构建目标:
delta_x  = x_grid - center_2d_x     # 像素到投影中心的 2D 偏移
delta_y  = y_grid - center_2d_y
depth    = GT_depth                 # 3D 中心深度
bbox_targets_3d = [delta_x, delta_y, depth, GT_w, GT_l, GT_h, sin(GT_yaw), GT_vx, GT_vy]
centerness_target = exp(-alpha * dist / stride)  # 基于距离的中心度
```

**各损失一览**：

| Loss 名 | 任务 | 损失函数 | 权重机制 | 权重值 |
|---------|------|----------|----------|--------|
| `loss_cls` | 类别分类 (10类) | Focal Loss (γ=2.0, α=0.25) | loss_weight | 1.0 |
| `loss_offset` | 2D 偏移 (Δx,Δy) | Smooth L1 (β=1/9) | code_weight | 1.0 |
| `loss_depth` | 深度 z | Smooth L1 (β=1/9) | code_weight | **0.2** |
| `loss_size` | 3D 尺寸 (w,l,h) | Smooth L1 (β=1/9) | code_weight | 1.0 |
| `loss_rotsin` | 朝向 sin(θ) | Smooth L1 (β=1/9) + sin差分 | code_weight | 1.0 |
| `loss_velo` | 速度 (vx,vy) | Smooth L1 (β=1/9) | code_weight | **0.05** |
| `loss_centerness` | 中心度 | BCE 交叉熵 | loss_weight | 1.0 |
| `loss_dir` | 方向二分类 (0/π) | 交叉熵 | loss_weight | 1.0 |
| `loss_attr` | 属性分类 (9类) | 交叉熵 | loss_weight | 1.0 |

**权重设计解读**：
- `depth` 权重 0.2 — 单目深度估计高度不确定，降权防止其噪声主导梯度
- `velo` 权重 0.05 — 速度预测同样极具挑战性，且 nuScenes 评分中速度指标权重本身就低
- 所有 `loss_weight` 默认 1.0，真正的差异化由 `code_weight` 实现

### 4.4 如何汇总 — 联合训练，一次反向传播

**FCOS3D 采用多任务联合训练，不是各自独立优化。**

```
loss_dict = {
    'loss_cls': ..., 'loss_offset': ..., 'loss_depth': ...,
    'loss_size': ..., 'loss_rotsin': ..., 'loss_velo': ...,
    'loss_centerness': ..., 'loss_dir': ..., 'loss_attr': ...
}
      │
      ▼ MMEngine 自动处理
total_loss = sum(loss_dict.values())   ← 所有 loss 直接求和
total_loss.backward()                  ← 一次反向传播
optimizer.step()                       ← 一次参数更新，更新所有参数
```

所有 9 个 loss 共享同一个 backbone + neck 的梯度信号，端到端训练。code_weight 通过 `bbox_weights` 直接乘到每个样本每个维度的 Smooth L1 loss 上（`fcos_mono3d_head.py:381-385`）：

```python
bbox_weights = ones(N_pos, 9)
bbox_weights = bbox_weights * tensor(code_weight)  # [1, 1, 0.2, 1, 1, 1, 1, 0.05, 0.05]

loss_offset = SmoothL1(pred[:, :2], target[:, :2], weight=bbox_weights[:, :2])
loss_depth  = SmoothL1(pred[:, 2],  target[:, 2],  weight=bbox_weights[:, 2])
# ...

# 分类损失在所有像素上计算 (正+负), avg_factor = num_pos + num_imgs
# 回归损失只在正样本上计算, avg_factor = num_pos
```

---

## 5. 推理流程：前处理 → 模型 → 后处理

### 5.1 前处理 (Preprocessing)

**代码**: `Det3DDataPreprocessor.forward()` → `collate_data()` (`data_preprocessor.py`)

```
输入: 原始图像 (H, W, 3) + meta 信息(cam2img 等)

  1. numpy/tensor → Tensor(3, H, W)
  2. BGR→RGB (bgr_to_rgb=True)
  3. Normalize: (img - mean) / std
     mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0]
  4. Pad 到 32 的倍数 (右下角补零)

输出: batch_inputs = {'imgs': Tensor(N, 3, H_pad, W_pad)}
      batch_data_samples (cam2img 等 meta 透传)
```

不涉及 cam2img 计算，仅透传 meta。

### 5.2 模型前向 (同训练，不使用 cam2img)

与训练时的 forward 完全一致，输出 2.5D 编码预测。

### 5.3 后处理 — 预测的"解码" (使用 cam2img 反投影)

**代码**: `FCOSMono3DHead.predict_by_feat()` → `_predict_by_feat_single()` (`fcos_mono3d_head.py:484-715`)

```
输入: 5层预测编码 + batch_img_metas (含 cam2img)

Step 1: 解码到图像空间
  ├─ sigmoid(cls_score), sigmoid(centerness)     → 概率值
  ├─ scale_depth(.)exp(), scale_size(.)exp()      → 正值深度/尺寸
  └─ offset × stride → 恢复到原图尺度
  → bbox_pred = [u, v, depth, w, l, h, sinθ, vx, vy]

Step 2: 计算 2D 中心位置
  └─ bbox_pred[:, :2] = grid_points - offset → (u, v) 图像坐标

Step 3: ✅ 反投影到相机坐标系
  └─ points_img2cam([u,v,depth], cam2img)
     X = (u - c_u) * z / f_u
     Y = (v - c_v) * z / f_v
     Z = z

Step 4: ✅ 局部 yaw → 全局 yaw
  └─ decode_yaw(bbox, centers2d, dir_cls, cam2img)
     方向二分类消除 θ vs θ+π 歧义
     加入射线方向: yaw = atan2(u-c_u, f_u) + dir_rot

Step 5: 3D NMS → 最终输出
  └─ nms_score = cls_score × centerness
     box3d_multiclass_nms → Top-K

输出: pred_instances_3d
  ├─ bboxes_3d: (K, 9) [x, y, z, w, l, h, yaw, vx, vy] (相机坐标系)
  ├─ scores_3d: (K,)
  └─ labels_3d: (K,)
```

---

## 6. 完整流程总结

```
┌──────────────────────────────────────────────────────────────────┐
│  训练阶段                                                         │
│                                                                   │
│  [3D GT] --cam2img投影--> [GT编码: centers_2d, depths, ...]     │
│                              │                                    │
│  [图像] --> CNN网络 --> [预测编码] --> {Loss 对比} --> total_loss │
│                                            │                      │
│                                    一次 backward 更新所有参数      │
│                                                                   │
│  ✅ cam2img 在"投影"步骤使用，网络前向和Loss计算不使用            │
├──────────────────────────────────────────────────────────────────┤
│  推理阶段                                                         │
│                                                                   │
│  [图像] --> CNN网络 --> [预测编码] --cam2img反投影--> [3D检测结果] │
│                                                                   │
│  ✅ cam2img 在"反投影"步骤使用，网络前向不使用                    │
└──────────────────────────────────────────────────────────────────┘
```

**核心范式**：前处理用 `cam2img` **投影** GT → 2.5D 编码 → 网络学习该编码 → 后处理用 `cam2img` **反投影**回 3D。网络本身始终不感知相机参数。

---

## 7. 相机参数说明

只需要 **相机内参 `cam2img`**（不需要外参）：

```
cam2img = [[f_u, 0,   c_u],
           [0,   f_v, c_v],
           [0,   0,   1  ]]
```

- `f_u, f_v`: 焦距（像素单位）
- `c_u, c_v`: 光心坐标

存储位置：`img_meta['cam2img']`，由 `Pack3DDetInputs` 自动包装。

---

## 8. 关键设计细节

### 8.1 Centerness（中心度）

预测每个像素到 GT 中心的归一化距离，用于抑制远离物体中心的低质量预测：

```python
centerness_target = exp(-centerness_alpha * dist / stride)  # alpha=2.5
```

代码: `fcos_mono3d_head.py:955`

### 8.2 方向分类器 (Direction Classifier)

朝向角 θ 有周期性歧义（θ 和 θ+π 在单目图像中看起来相似），引入二分类器：
- bin 0 / bin 1: 朝向在 [0, π) 或 [π, 2π)
- `dir_offset = π/4` 让 bin 边界避开车头正对相机的情况

代码: `fcos_mono3d_head.py:232-267`

### 8.3 sin 差分 (diff_rad_by_sin)

计算旋转 loss 时用 sin 差而非直接角度差，避免角度周期性问题：

```python
sin_diff = sin(θ_pred) * cos(θ_gt) - cos(θ_pred) * sin(θ_gt)
```

代码: `fcos_mono3d_head.py:207-230`

### 8.4 Center Sampling

只有落入 GT 中心附近（`center_sample_radius=1.5 × stride`）的像素才被分配为正样本，减少物体边缘处的不确定性。

### 8.5 可学习 Scale 参数

每层 FPN 有独立可学习 scale 参数 (`Scale(1.0)`)，用于自动调整 offset、depth、size 预测的尺度。

---

## 9. 使用方式

```python
from mmdet3d.apis import init_model, inference_detector

config_file = 'configs/fcos3d/fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d.py'
checkpoint_file = 'fcos3d_r101-caffe-dcn_fpn_head-gn_8xb2-1x_nus-mono3d.pth'

model = init_model(config_file, checkpoint_file, device='cuda:0')
result = inference_detector(model, img)
```

---

## 10. 性能参考

nuScenes test set（2021年提交时）：NDS ~0.37, mAP ~0.30。PGD 论文在此基础上提升 NDS 至 ~0.41。

---

## 11. 继承关系

```
mmdet.SingleStageDetector
    └─ SingleStageMono3DDetector     ← 单目3D检测器基类
        └─ FCOSMono3D                ← 本模型

mmdet.models.dense_heads.BaseDenseHead
    └─ BaseMono3DDenseHead
        └─ AnchorFreeMono3DHead      ← anchor-free单目3D头基类
            └─ FCOSMono3DHead        ← 本模型 Head
                └─ PGDHead           ← PGD 改进版
```
