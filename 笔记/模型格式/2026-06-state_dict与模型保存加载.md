> **归档来源**：`老笔记/暂时未归纳点6月.md` 第 424–518 行（按主题拆分归档）

---

	描述：模型结构（代码定义的nn.Module类）与state_dict（OrderedDict[str, Tensor]形式的权重数据）共同构成完整模型。.pt文件本质就是state_dict，里面只有参数数值（如"layer1.weight"→Tensor），没有模型结构（类/forward/计算图）。模型结构来自代码里的model = MyModel()，这创建了nn.Module及每层的Parameter（weight/bias）和buffer（BN统计量），这些才是forward时真正参与计算的东西。load_state_dict不做"替换参数对象"，而是遍历model.named_parameters()/named_buffers()，用key去ckpt里找对应值，执行param.data.copy_(ckpt[name])把数值填进去。forword运行时只用self.layer.weight等模型内部Tensor，不会访问state_dict。strict=True时key必须完全一致否则报错中断，strict=False则能匹配就加载、不匹配忽略（但shape不一致仍报错）。完整生命周期：保存（model.state_dict()→torch.save→.pt文件）→加载（model=MyModel()先建结构→torch.load读参数→model.load_state_dict填进去）。类比：模型结构=已组装好的机器，state_dict=带标签的零件包，load_state_dict=按标签把零件装进机器，forward=机器运行。
	冲突：
	    ① 容易以为.pt文件里包含模型结构——实际上.pt只有state_dict数据，没有类定义/forward/计算图，加载时必须先用代码创建模型结构
	    ② 容易以为load_state_dict是替换参数对象——实际上是param.data.copy_()把ckpt的值拷贝进已有Parameter的.data，参数对象本身没被替换
	    ③ 容易以为forward时用的是state_dict里的Tensor——实际上forward用的是model内部注册的Parameter（如self.layer.weight），state_dict只是保存/加载时的"数据快照"
	    ④ 容易把"模型结构以state_dict接收"说成"结构接收数据"——更严谨的说法是：模型结构用state_dict来填充参数，state_dict不定义结构
	    ⑤ 容易忽略strict=False时shape不匹配仍然报错——strict=False只放宽key匹配，shape不一致永远会报错
	    ⑥ 容易忽略buffer（如BN的running_mean/running_var）也包含在state_dict中——state_dict不只包含Parameter，还包括buffer，model.named_buffers()也在拷贝范围内
	解决方法：
	    （1）state_dict的本质
	        state_dict = OrderedDict[str, Tensor]
	        来源：model.state_dict()遍历所有nn.Module的参数和buffer，生成"模块路径字符串→Tensor数值"的映射
	        内容示例："backbone.layer1.conv1.weight" → Tensor(64,3,7,7), "backbone.layer1.bn1.running_mean" → Tensor(64)
	        关键认知：state_dict是一个"参数名→参数值"的平面字典，没有任何层级结构信息
	    （2）模型结构从哪里来
	        model = MyModel() 创建了：
	            nn.Module层级树（如 model.backbone.layer1.conv1）
	            每层的nn.Parameter（weight/bias，requires_grad=True，参与训练）
	            每层的buffer（running_mean/running_var等，requires_grad=False，保存但不需要梯度）
	        这些Parameter和buffer的shape在代码初始化时已确定，state_dict必须与之匹配
	    （3）load_state_dict的内部流程
	        Step 1——获取目标：model.named_parameters() + model.named_buffers() → 模型内部的完整参数名列表
	        Step 2——匹配：对每个模型参数名name，在ckpt中查找ckpt[name]
	        Step 3——校验shape：target_param.shape == ckpt[name].shape，不匹配则报错（即使strict=False）
	        Step 4——拷贝：target_param.data.copy_(ckpt[name])，把ckpt的数值复制到模型参数的.data中
	        关键：copy_()是in-place操作，不创建新Tensor，不改变Parameter对象的引用，只改其内部数值
	    （4）forward运行时到底用哪个Tensor
	        模型内部的Parameter是注册在nn.Module上的：self.layer.weight（这是一个nn.Parameter，即Tensor子类）
	        forward中执行的是：output = F.conv2d(x, self.layer.weight, self.layer.bias)
	        用的是self.layer.weight这个Tensor对象——与state_dict中的key同名、同引用，但不是"去字典里查"
	        加载后关系：self.layer.weight.data 的数值 == ckpt["layer.weight"] 的数值（因为copy_过了）
	        如果加载失败（没调用load_state_dict或key不匹配）：self.layer.weight保持初始化的随机值
	    （5）strict参数的精确行为
	        strict=True（默认）：
	            检查missing keys（模型有但ckpt没有）→ 有则报错
	            检查unexpected keys（ckpt有但模型没有）→ 有则报错
	            全部匹配才成功加载
	        strict=False：
	            missing keys → 忽略，该参数保持初始化的值（随机或预定义）
	            unexpected keys → 忽略，不加载
	            shape不匹配 → 仍然报错（无论strict值）
	        使用场景：迁移学习（去掉分类头）、模型微调（只加载部分层）、模型结构有微小变化
	    （6）完整生命周期（保存→加载→推理）
	        保存阶段：
	            torch.save(model.state_dict(), "model.pt")
	            → 得到一个"参数字典文件"，可以跨机器、跨时间传输
	        加载阶段：
	            model = MyModel()                  # ①用代码创建结构（必须有类定义）
	            ckpt = torch.load("model.pt")      # ②反序列化得到OrderedDict
	            model.load_state_dict(ckpt)        # ③按名匹配+拷贝数据
	            model.eval() / model.to(device)    # ④设置模式和设备
	        推理阶段：
	            output = model(x)                  # forward只用模型内部参数
	        关键：①和②的顺序不能颠倒——必须先有结构才能填参数
	    （7）如果只有.pt文件，怎么反推模型结构
	        读取key规律：
	            ckpt.keys() → 看参数名模式（如"backbone.layer1.0.conv1.weight"暗示ResNet结构）
	        从shape反推：
	            卷积层weight shape = (out_ch, in_ch, k, k) → 可知通道数和卷积核大小
	            线性层weight shape = (out_dim, in_dim) → 可知输入输出维度
	            嵌入层weight shape = (vocab_size, embed_dim) → 可知词表和维度
	        局限性：只能推断参数shape，无法知道forward逻辑（如用了什么激活函数、有没有残差连接、attention的细节）
	        经验：如果能找到作者公开的模型代码 → 直接对应；如果没有 → 需要根据shape和常见的模型模式手动构建
	    （8）三个角色各司其职
	        角色              | 来源                                    | 内容
	        模型结构            | model = MyModel() 代码创建               | nn.Module 层级树、Parameter（weight/bias）、buffer（BN统计量）
	        state_dict        | model.state_dict() 或 .pt 文件           | 参数名→Tensor 数值的平面字典
	        torch.save/load   | torch.save(sd, "model.pt") / sd = torch.load("model.pt") | 字典序列化到磁盘 / 从磁盘反序列化回字典形态
	        load_state_dict   | model.load_state_dict(sd)                | 按名匹配 + copy_() 拷贝数据到模型内部 Parameter
	    补充：Parameter → state_dict → .pt 的层级关系
	        Parameter (nn.Parameter) 是权重的"容器"，存储在 GPU/CPU 上，是实际参与 forward 计算的 Tensor
	            ↓ model.state_dict()
	        state_dict (OrderedDict) 是权重的"传递形态"，扁平字典，key=路径，value=Tensor 数值
	            ↓ torch.save(sd, "model.pt")
	        .pt 文件 是权重的"存储形态"，磁盘上的序列化二进制，不含模型结构
	            ↓ sd = torch.load("model.pt")
	        state_dict 回到内存字典形态
	            ↓ model.load_state_dict(sd)
	        Parameter 的 .data 被 copy_() 填入加载的数值
	        核心认知：Parameter 始终是权重的"容器"，state_dict 是"传递形态"，.pt 是"存储形态"。torch.load 负责反序列化（磁盘→内存），load_state_dict 负责匹配灌入（字典→模型）。
	    （9）终极类比表
	        概念              | 类比
	        模型结构            | 机器（已经组装好框架，包含齿轮、传动等结构）
	        state_dict        | 零件包（每个零件带标签，如"齿轮A"→具体尺寸数值）
	        load_state_dict   | 按标签把零件装进机器对应位置
	        forward           | 机器通电运行
	        .pt文件            | 零件包被装在密封箱子里快递（只有零件，没有装配图纸）
	        模型初始化          | 先造一台机器框架（齿轮是随机的毛坯）
	        strict=True       | 零件标签必须和机器位置完全一一对应
	        strict=False      | 有标签对得上的就装，对不上的跳过
	    （10）一句终极总结
	        结构靠代码定义，权重靠state_dict传递，torch.save/load负责磁盘持久化，load_state_dict负责"按名匹配+拷贝数据"，forward只用模型内部参数不用state_dict——四者（代码结构、state_dict、torch.save/load、load_state_dict）缺一不可，共同构成PyTorch模型的完整生命周期。

	6、
	所属：深度学习框架 推理加速 ONNX TensorRT 模型优化
