> **归档来源**：`老笔记/暂时未归纳点6月.md` 第 519–603 行（按主题拆分归档）

---

	描述：ONNX是中间表示（IR），是一种"标准计算图格式"，它不是执行引擎，不专注极致性能，核心是通用交换格式。TensorRT是高性能推理编译器+执行引擎，会对模型做：Kernel Fusion（如MatMul+Add+ReLU熔成1个kernel减少显存IO）+ Kernel Auto-tuning（尝试多种CUDA实现选最快，如不同block size/tile strategy）+ 内存优化（内存复用、中间tensor不落地提高cache命中）+ 精度优化（FP32→FP16/INT8自动量化）+ 常量折叠 + 更激进的Layer Fusion（Conv+BN+ReLU甚至部分Attention block）+ 动态shape优化。TensorRT快不只是因为fusion，而是"面向GPU的编译器"做了fusion+kernel选择+内存优化+精度优化的组合拳；四招分别打在不同的瓶颈维度上——融合减IO、调优减计算、内存复用降峰值、降精度同时减计算/显存/搬运，且彼此交叉放大。PyTorch→ONNX导出时默认基本不做强优化，重点在保持语义正确而非激进优化。ONNX Runtime（非ONNX本身）配合CUDA/TensorRT EP也能做一些fusion和kernel优化，但不如TensorRT激进。类比：ONNX=标准图纸，TensorRT=根据GPU定制的高性能机器。
	冲突：
	    ① 容易以为"TensorRT快只是因为算子融合（fusion）"——实际上fusion只是原因之一，kernel auto-tuning和内存优化同样关键甚至更重要
	    ② 容易以为"ONNX导出时会自动做fusion优化"——实际上PyTorch→ONNX导出默认不做激进优化，只保持语义正确性
	    ③ 容易把ONNX和ONNX Runtime混淆——ONNX只是图格式（IR），不执行推理；ONNX Runtime才是执行引擎，能做一定优化
	    ④ 容易以为"ONNX如果也做fusion就会和TensorRT一样快"——实际上TensorRT的kernel auto-tuning、内存复用、精度优化等能力远超ONNX Runtime，即使都做fusion差距仍然很大
	    ⑤ 容易忽略Kernel Auto-tuning的重要性——同一个算子（如MatMul）在不同shape/GPU上有多种CUDA实现，TensorRT会benchmark后选最优，这是手动写代码做不到的
	    ⑥ 容易忽略内存优化——中间tensor的内存复用和避免落地（kernel fusion的延伸）对实际推理速度影响巨大，尤其在显存紧张时
	    ⑦ 容易把精度优化只理解成"算得快"——实际上降精度是"算力、显存、显存带宽"三赢：位宽变小后算得更快、占用显存更少、需要搬运的数据量也同步变小
	    ⑧ 容易把四招当成各自独立的技巧——实际上它们是互相放大的组合拳，单独看会低估收益，见（9）
	解决方法：
	    （1）ONNX的本质
	        ONNX = Open Neural Network Exchange = 开放的神经网络交换格式
	        内容：一个有向无环图（DAG），节点是算子（Conv、MatMul、Add等），边是tensor
	        目的：跨框架互操作——PyTorch训练 → ONNX → TensorRT/ONNX Runtime/OpenVINO推理
	        它不是推理引擎，只是一个"标准文件格式"
	        PyTorch导出ONNX：torch.onnx.export(model, dummy_input, "model.onnx") → trace或script → 生成ONNX graph
	        导出时做的优化：非常有限，基本是1对1映射（PyTorch op→ONNX op），不改变计算结构
	    （2）TensorRT的本质
	        TensorRT = NVIDIA官方的"深度学习推理编译器+运行时"
	        输入：ONNX（或TensorFlow/Caffe/直接API）
	        输出：针对具体GPU型号优化过的"执行计划"（engine/.plan文件）
	        编译过程：解析ONNX graph → 图优化（fusion/消除冗余）→ kernel selection（auto-tuning）→ 内存规划 → 生成序列化engine
	        运行时：加载engine → 分配内存 → 执行 → 输出结果
	        关键认知：TensorRT是一个编译器，不是简单的运行时，编译阶段的优化决定了推理性能的上限
	    （3）Kernel Fusion——省的是"搬运时间"（减IO）
	        未融合：MatMul → 写结果到HBM → Add → 读回 → 写结果到HBM → ReLU → 读回 → 写结果到HBM
	            每两个算子之间都要把中间结果写回显存、再读出来，这几趟往返是纯浪费
	        融合后：MatMul + Add + ReLU在一个kernel中完成 → 中间结果只留在寄存器/shared memory → 只写一次HBM
	        关键瓶颈：GPU算力极强，真正的瓶颈是"把数据从显存（HBM）搬进计算单元（Cache/寄存器），算完再搬回去"
	            → 融合的本质是"只搬一次，算完一气呵成"，省下的是IO而不是计算
	        常见融合模式：
	            Conv + BN + ReLU（经典，几乎所有框架都做）
	            MatMul + BiasAdd + Activation
	            Multi-Head Attention的部分计算（如QKV投影的融合）
	    （4）Kernel Auto-tuning——省的是"纯计算时间"
	        问题背景：同一个算子（如卷积）在不同shape（H,W,C,K）和不同GPU（A100/V100/T4）上有多种实现策略——Winograd/FFT/im2col+GEMM/direct等，block_size/tile_size也有多种选择
	        TensorRT的做法：编译时为每个算子尝试多种kernel实现 → 跑micro-benchmark → 选该GPU+该shape下"计算步骤最少、最契合当前GPU架构"的实现
	        收益：直接缩短纯计算耗时
	        为什么PyTorch做不到：PyTorch是JIT执行，无法在运行前做全局搜索；cuDNN heuristics是快速但粗糙的选择
	        类比：PyTorch（cuDNN heuristic）=凭经验快速点菜；TensorRT=把候选菜都实测一遍再选最好的
	    （5）内存优化——省显存峰值，也隐藏地省搬运
	        机制1——内存复用：
	            推理时大量中间tensor的生命周期很短 → TensorRT分析每个tensor的活跃区间 → 不重叠的tensor共用同一块显存
	            收益1（省显存）：显存峰值大幅降低，而OOM（显存溢出）往往就是峰值撑爆
	            收益2（隐藏的省搬运）：内存被复用、整理得更紧凑后，数据在物理距离上更近，取数路径更短 → L2 cache命中率提升 → 数据只要在缓存里就不用去遥远的HBM取，时间自然省下来
	            类比：把仓库整理得井井有条，快递员取货的往返距离变短
	        机制2——避免中间结果落地：
	            fusion的延伸——不仅是多个算子合成一个kernel，更关键的是中间结果不写回HBM（高带宽显存），只在寄存器/shared memory中传递
	        实际影响：对于大模型（如LLM），内存优化带来的加速可能比fusion更显著
	    （6）精度优化——FP16和INT8的"三重暴击"
	        FP16（半精度）：
	            大部分现代GPU（V100/A100）有专用的FP16 Tensor Core → FP16吞吐是FP32的2~8倍
	            TensorRT自动插入类型转换节点，保证数值稳定
	            策略：敏感层（如最后输出）保持FP32，中间层用FP16
	        INT8（量化）：
	            需要校准（calibration）：跑少量数据统计每层的数值范围 → 确定缩放因子 → 权重和激活值都量化为INT8
	            代价：有精度损失（通常很小），需要校准数据
	        降精度为什么是"三重暴击"（三个维度同时受益）：
	            ① 算得快：现代GPU（尤其带Tensor Core的）对FP16/INT8有专门硬件加速电路，算力是FP32的数倍
	            ② 省显存：32位浮点变成16位/8位，占用直接减半甚至降到1/4（FP32→INT8），能塞下更大的模型
	            ③ 搬得快：数据量变小，需要从HBM搬到计算单元的"货量"也变小；同样带宽下搬运时间减半 → 反过来又缓解了内存带宽瓶颈
	        一句话：FP32→INT8 = 计算、显存、搬运三赢；FP16是相对保守的中间档
	    （7）PyTorch → ONNX → TensorRT的完整链路和每步做了什么
	        Step 1——PyTorch模型（model.py）：
	            训练好的模型，包含代码结构+state_dict
	            此时是动态图（eager mode），每次forward都构建计算图
	        Step 2——导出ONNX（torch.onnx.export）：
	            做了什么：trace（记录一次实际forward的计算路线）→ 每个PyTorch op映射为等价的ONNX op → 保存为静态图
	            没做什么：不做fusion、不做kernel选择、不做内存优化
	            结果：model.onnx（静态计算图文件）
	        Step 3——TensorRT编译（trtexec / TensorRT API）：
	            做了什么：
	                ① 解析ONNX graph
	                ② 图优化：消除无用节点、常量折叠、算子融合（水平融合+垂直融合）
	                ③ Kernel selection：为每个融合后的算子选最优CUDA kernel
	                ④ 内存规划：分配buffer、复用不重叠的内存区域
	                ⑤ 序列化：生成engine（.plan文件）
	            结果：model.plan（GPU专用优化后的执行计划）
	        Step 4——TensorRT推理（TensorRT Runtime）：
	            加载engine → 创建execution context → 分配IO buffer → enqueue任务到CUDA stream → 执行
	    （8）为什么ONNX Runtime不如TensorRT（即使两者都做fusion）
	        架构差异：
	            TensorRT：编译期做全图分析 → 全局优化（跨层fusion、全局内存规划）
	            ONNX Runtime：更多是"在线优化"——在加载图时做局部优化，缺少全局视角
	        Kernel质量：
	            TensorRT：针对NVIDIA GPU深度定制，kernel由NVIDIA工程师手工调优，用到了各代GPU的专有指令
	            ONNX Runtime：使用cuDNN/cuBLAS等通用库，不如TensorRT kernel针对性强
	        Auto-tuning：
	            TensorRT有，ONNX Runtime基本不做
	        一句话：TensorRT是NVIDIA自家产品，了解GPU底层的每一个细节；ONNX Runtime是跨硬件通用方案，优化相对保守
	    （9）四招如何交叉放大——组合拳的收益不是简单相加
	        一招一维，先看各自打在哪：
	            Kernel Fusion      → 减IO（少搬数据）
	            Kernel Auto-tuning → 减计算（选最快的实现）
	            内存优化           → 减显存峰值（并间接提高cache命中、减IO）
	            精度优化           → 减计算 + 减显存 + 减搬运（三赢）
	        交叉收益（1+1>2 的地方）：
	            ① 精度 × 搬运：位宽减半/降到1/4后，同样的显存带宽能搬更多数据 → 带宽瓶颈被动缓解
	            ② 内存 × IO：复用让内存布局更紧凑 → L2 cache命中率上升 → 相当于又少搬了一部分数据
	            ③ 融合 × 内存：融合把中间结果留在片上，配合内存规划才能真正做到"中间tensor不落地"
	            ④ 调优 × 一切：kernel选得越快，省下来的时间才不会被庞大的数据搬运吃掉
	        反例（拆开看会低估或失效）：
	            没有内存优化，融合后仍可能因为显存不够而频繁交换
	            没有精度优化，自动调优选出的kernel再好，也会被庞大的搬运量拖慢
	        一句话闭环：融合减IO，调优减计算，内存复用减峰值，降精度实现计算/显存/搬运三赢
	    （10）一句话终极总结
	        ONNX是一张"标准图纸"，TensorRT是根据这张图纸、针对具体GPU型号打造的"定制高性能机器"——它快不是单一原因，而是fusion（减IO）+ auto-tuning（选最优kernel）+ 内存优化（复用+不落地）+ 精度优化（FP16/INT8）的四重组合拳。
