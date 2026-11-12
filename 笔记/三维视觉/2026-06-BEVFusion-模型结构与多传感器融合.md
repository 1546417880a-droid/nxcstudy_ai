> **归档来源**：`老笔记/暂时未归纳点6月.md` 第 688–786 行（按主题拆分归档）

---

		描述：BEVFusion是MIT韩松组提出的多传感器（相机+LiDAR+Radar）多任务（检测+BEV语义分割）融合框架。与BEVFormer的核心区别：BEVFormer是Encoder-Decoder固定Q形式（query主动检索），BEVFusion是纯Encoder密集Q形式（每个BEV格子就是一个Q，密集预测）。BEV构建方式：相机端通过DepthLSSTransform（深度估计 + Lift-Splat投影）把图像特征"撒"到BEV栅格，LiDAR端通过SparseEncoder（3D稀疏卷积）后压扁Z轴形成BEV特征。融合方式：ConvFuser（concat后3x3卷积）或AddFuser（各模态对齐通道后相加，训练时随机丢弃某模态以增加鲁棒性）。融合后的BEV特征再经过Decoder Backbone（2D CNN）+ Neck做精炼，最后接多任务Head：TransFusionHead/CenterHead（3D检测）+ Map Segm Head（BEV语义分割）。全程无object query，无时序注意力，是密集预测风格。
		冲突：
		    ① 容易把BEVFusion的"Decoder"和Transformer Decoder混淆——BEVFusion的decoder是一个2D CNN backbone+neck在BEV特征上做精炼，不是Transformer decoder，没有cross-attention和object query
		    ② 容易以为所有BEV方法都用query式——BEVFusion/PointPillar/CenterPoint都是纯Encoder密集预测，只有DETR/BEVFormer/PETR是query式
		    ③ 容易以为融合就是用Transformer做attention——BEVFusion的融合就是简单的concat+Conv或add，没有attention机制
		    ④ 容易忽略AddFuser的随机dropout机制——训练时随机丢弃一个模态的BEV特征（模拟传感器失效），测试时不丢，这是BEVFusion鲁棒性的关键设计
		    ⑤ 容易混淆BEVFusion和BEVFormer的BEV构建方式——BEVFusion是"投影+堆积"（深度估计→撒到BEV格子），BEVFormer是"BEV query主动检索"（跨视角attention），前者快但依赖深度质量，后者精度高但慢
		解决方法：
		    （1）整体架构（从多传感器到多任务输出）
		        输入：多相机图像(6 views) + LiDAR点云 + [可选] Radar点云
		        ├── Camera Encoder:
		        │   SwinT/ResNet(backbone) → GeneralizedLSSFPN(neck) → DepthLSSTransform(vtransform)
		        │   └── 深度估计分支预测每个像素深度分布
		        │   └── Lift-Splat: 特征×深度分布 → 3D视锥 → 撒到BEV栅格
		        │   └── Camera BEV: (B, 80, 128, 128)
		        ├── LiDAR Encoder:
		        │   Voxelization(体素化) → SparseEncoder(3D稀疏卷积 SubMConv3d+SparseConv3d)
		        │   └── 压缩Z轴 → to_dense
		        │   └── LiDAR BEV: (B, 128, 128, 128)
		        ├── [可选] Radar Encoder:
		        │   Voxelization → PillarEncoder
		        │   └── Radar BEV: (B, C, H, W)
		        ├── Fuser (多模态融合):
		        │   ConvFuser: cat(cam_bev, lidar_bev) → Conv2d → BN → ReLU
		        │   AddFuser: transform_cam(cam_bev) + transform_lidar(lidar_bev)，训练时随机drop一个模态
		        ├── Decoder (BEV特征精炼——注意不是Transformer Decoder！):
		        │   Backbone(2D CNN如SECONDFPN) → Neck(FPN)
		        └── Heads (多任务):
		            ├── Detection: TransFusionHead(Transformer式query) / CenterHead(heatmap密集)
		            └── Segmentation: Map Segm Head(逐像素分类)
		    （2）DepthLSSTransform（相机→BEV的核心）
		        输入：图像特征 (B, N, C, H, W)
		        步骤：
		            ① 深度估计：深度网络输出每个像素的深度分布 (D bins, 如1~60m分成118个bin)
		            ② 外积：每个像素特征(1×C)  ⊗  深度分布(D×1) → 视锥特征(D, C)
		            ③ 坐标变换：每个视锥点根据相机内外参映射到3D世界坐标
		            ④ 体素化：将3D视锥点按(x,y)分配到BEV栅格，同一格子的多个点做sum pooling
		        输出：Camera BEV (B, C, H_bev, W_bev)
		        关键认知：这不是query式检索，而是"每个像素根据自己的深度信念，把特征撒到对应的3D位置"
		        与BEVFormer的区别：BEVFormer是BEV query去图像里"找"信息，LSS是图像主动"投"信息到BEV
		    （3）SparseEncoder（LiDAR→BEV的核心）
		        输入：体素化后的稀疏点云 (feats, coords)
		        步骤：
		            ① SubMConv3d: 在非空体素上做3D卷积（保持稀疏性）
		            ② SparseConv3d: stride=2逐步降采样空间分辨率
		            ③ 4个stage后得到 (B, 128, X, Y, Z) 的3D稀疏特征
		            ④ 压缩Z轴 → 变成 BEV格式: (B, 128, H_bev, W_bev)
		        关键：稀疏卷积只在有点云的体素上计算，大量空体素不占用计算和显存
		    （4）两种Fuser对比
		        维度          | ConvFuser                          | AddFuser
		        融合方式        | concat后接3x3 Conv                  | 各模态先1x1对齐通道，再相加
		        参数量          | 较大(sum(in_channels)×out_channels) | 较小(各模态独立1x1)
		        训练鲁棒性       | 无特殊处理                          | 随机drop一个模态（模拟传感器失效）
		        推理            | 正常融合                            | 正常融合
		        AddFuser的关键设计：训练时random.random()<dropout → 随机丢一个模态的BEV特征（权重=0）
		        → 模型学会"只有一个传感器也能检测" → 推理时即使LiDAR坏了也能靠相机继续工作
		    （5）TransFusionHead vs CenterHead（两种检测Head）
		        CenterHead:
		            类型：anchor-free密集检测
		            输入：BEV特征图 (B, C, H, W)
		            输出：每个BEV位置预测 → heatmap(中心置信度) + box参数
		            后处理：top-K取峰值位置 → NMS去重
		            特点：快，传统方法，需要NMS
		        TransFusionHead:
		            类型：Transformer式query检测（小Decoder）
		            输入：BEV特征 + 初始proposals(来自heatmap)
		            过程：N_proposals个query → self-attn → cross-attn(BEV) → FFN → 输出bbox
		            特点：精度更高，有query但不同于BEVFormer的大Decoder
		        注意：TransFusionHead的query数量很少(如200个)，且来自heatmap初始化，不是纯可学习的
		    （6）Map Segmentation Head（BEV语义分割）
		        输入：融合后的BEV特征 (B, C, H_bev, W_bev)
		        结构：几层Conv → 逐像素分类
		        输出类别：车道线、可行驶区域、人行横道等
		        特点：纯密集预测（每个BEV格子一个分类结果），完全无query
		    （7）BEVFusion vs BEVFormer 完整对比
		        维度              | BEVFormer                         | BEVFusion
		        Q形式              | 可学习BEV query + object query     | BEV栅格密集Q（每个格子一个Q）
		        架构类型            | Encoder-Decoder，固定Q             | 纯Encoder（全部密集预测）
		        BEV构建方式         | Q通过cross-attn主动检索图像特征    | 深度估计+Lift-Splat投影 + 体素化
		        时序建模            | ✅ TSA（时序自注意力）              | ❌ 无
		        融合方式            | 无（纯视觉）                       | ConvFuser / AddFuser
		        传感器            | 只要相机                           | 相机 + LiDAR + Radar
		        检测方式            | DETR式query（大Decoder）            | TransFusionHead(小Decoder) / CenterHead
		        分割任务            | 无                                | ✅ BEV语义分割
		        是否需要NMS        | 不需要（Hungarian Matching）         | CenterHead需要，TransFusionHead不需要
		        核心优势            | 时序理解强，纯视觉端到端            | 多传感器融合精度高，多任务
		    （8）用"Q分类法"理解三种BEV方法
		        纯Encoder密集Q:
		            BEVFusion: 每个BEV格子=一个Q → 密集预测 → 检测head+分割head
		            CenterPoint: 每个BEV格子=一个Q → heatmap找中心 → 预测box
		        Encoder-Decoder固定Q:
		            BEVFormer: BEV query(固定) → BEV特征 → object query(固定) → 预测box
		            DETR/PETR: object query(固定) → cross-attn(图像特征) → 直接出box
		        关键区分：是否有"外部的可学习query"——有=固定Q(DETR系)，无=密集Q(CNN系)
		    （9）一句话总结
		        BEVFusion = 相机(LSS投影) + LiDAR(稀疏卷积) → 都在BEV空间中对齐 → 简单拼接/相加融合 → CNN精炼 → 多任务密集预测，全流程无object query、无时序注意力，是纯Encoder架构。与BEVFormer的query式检索形成"密集预测 vs query检索"的两种范式对立。

	9、
	所属：计算机视觉 3D目标检测 BEVFusion 密集预测范式 与 Query-based方法对比
