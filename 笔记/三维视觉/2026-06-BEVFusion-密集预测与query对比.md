> **归档来源**：`老笔记/暂时未归纳点6月.md` 第 787–830 行（按主题拆分归档）

---

	描述：BEVFusion / BEV CNN流派的本质是密集预测——融合后的BEV特征是一个稠密空间特征图(B,C,H,W)，同一个共享参数的Conv/MLP Head在每个BEV网格位置滑动，对每个grid的特征向量做相同映射，输出类别+位置+属性。数学上：output[i,j] = f(feature[i,j])，其中f(·)对所有空间位置共享权重。因为BEV空间≈真实物理空间，所以"一个grid≈一个位置"，可以像语义分割一样做检测。核心区别：BEVFusion是"H×W个位置每个都预测→可能有冗余→需要NMS（部分方法）"，而BEVFormer是"固定N个query→每个query=一个物体→不密集→不需要NMS"。
	冲突：
	    ① 容易混淆"密集预测"和"每个grid一个独立模型"——实际上Head是共享参数的Conv，一个模型在整个BEV上滑动，不是H×W个模型
	    ② 容易误以为BEVFusion所有Head都是纯密集——实际上TransFusionHead是混合型：BEV feature→选出top-K（heatmap）→生成query→Transformer refine，是先密集后稀疏
	    ③ 容易误以为密集预测不需要NMS——实际上密集预测每个grid独立预测，多个grid可能响应同一物体，通常需要NMS去重；而query-based方法靠Hungarian Matching天然不重复
	解决方法：
	    （1）标准计算流程
	        BEV Feature (B, C, H, W)
	                ↓
	        共享 Encoder（CNN精炼）
	                ↓
	        统一 BEV Feature
	                ↓
	        Task Head（共享参数 Conv/MLP）
	                ↓
	        每个 grid 输出预测[class, bbox, velocity, ...]
	    （2）核心确认（逐条）
	        ＜所属：＞通用特征输入到任务头——Head只把语义→转成任务输出
	        ＜所属：＞每个网格一个向量——(B,C,H,W)，每个grid一个C维向量(如128/256)
	        ＜所属：＞都会经过同一个任务头——同一个Conv/MLP权重，共享函数f(·)
	        ＜所属：＞所以是密集预测——对每一个空间位置都做预测
	    （3）为什么叫"共享头"很重要
	        ❌ 不是每个grid一个模型
	        ✅ 一个模型(Head)在整个BEV上滑动 = Conv2d(kernel=1 or 3)
	    （4）BEVFusion vs BEVFormer 范式对比
	        | 维度 | BEVFusion（密集） | BEVFormer（query） |
	        |---|---|---|
	        | 预测方式 | H×W个位置，每个都预测 | 固定N个query（如900），每个=一个物体 |
	        | 冗余 | 可能有 | 无 |
	        | 需要NMS | 部分方法需要 | 不需要（Hungarian Matching） |
	        | 范式 | 密集预测 | query检索 |
	        | Head类型 | CenterHead（纯密集）/ TransFusionHead（密集→稀疏） | Deformable DETR Head |
	    （5）TransFusionHead = 混合型（重要补充）
	        BEV feature → 选出top-K（heatmap峰值）→ 生成query → Transformer refine
	        不是完全密集，而是"密集→稀疏(query)"的桥梁
	    （6）为什么密集预测在BEV上成立
	        BEV空间 ≈ 真实物理空间 → 一个grid ≈ 一个位置 → 可以像语义分割一样做检测
	        这跟图像空间做检测不同——图像有透视畸变，BEV是正交投影
	    （7）一句话总结
	        在BEVFusion中，融合后的BEV特征是一个稠密空间特征图(B,C,H,W)，任务头通常采用共享参数的卷积结构，对每一个BEV网格位置的特征向量进行相同的映射输出类别+位置，本质上属于密集预测。与BEVFormer的query式检索形成两种范式对立。


		10、
		所属：计算机视觉 3D目标检测 BEV表示 CNN vs Transformer 架构选择
