> **归档来源**：`老笔记/暂时未归纳点6月.md` 第 831–898 行（按主题拆分归档）

---

		描述：在BEV空间上做检测/分割时，CNN通常比纯Transformer（self-attention）更稳定、收敛更快，根本原因在于BEV是规则物理空间网格（每个grid对应真实世界固定位置x,y），具有强局部连续性和空间一致性，而CNN的局部感受野和平移不变性（local inductive bias）与这种空间结构天然匹配，Transformer缺乏这种归纳偏置，需要从数据中自己"学会"什么是局部、什么是空间结构，导致训练更难、对数据更敏感、收敛更慢。此外，Transformer在200×200=40000个grid上的全局自注意力复杂度为O(N²)=1.6B，必须靠降采样/稀疏attention/deformable attention来近似，引入额外的近似误差和不稳定性。CNN还有一个工程优势：Conv2d在cuDNN/TensorRT上高度优化，计算图固定、延迟稳定；而Transformer的attention是memory-bound，推理优化难度更大。BEVFormer能work恰恰是因为它用Deformable Attention（只采样少量关键点）+ BEV query（自带空间含义）给Transformer"加了一点CNN的bias"，本质上是在两者之间取折中。
		冲突：
		    ① 容易以为Transformer一定比CNN强——在BEV这种规则物理空间上，CNN的归纳偏置是优势而非劣势，Transformer的"全局建模能力"在这里反而成为"噪音来源"
		    ② 容易忽略BEV的空间连续性——相邻grid对应相邻物理位置，一辆车占据多个连续grid，这种连续性CNN天然支持、Transformer需要自己学
		    ③ 容易以为attention的"全局视野"在BEV上总能提升精度——实际上初期训练时attention几乎是均匀分布，全局噪声淹没了局部信号
		    ④ 容易把BEVFusion的decoder理解成Transformer decoder——BEVFusion的decoder是2D CNN，正是利用了这一稳定性优势
		    ⑤ 容易忽略工程部署的差异——CNN在GPU上kernel高度优化、延迟可预测，Transformer的attention动态性导致延迟抖动和内存占用不稳定
		解决方法：
		    （1）BEV空间的本质：规则物理网格
		        BEV特征：(B, C, H, W)，每个grid对应真实世界一个(x,y)位置
		        关键性质：
		            - 空间连续：相邻grid对应相邻物理位置 → 局部相关性极强
		            - 平移不变：同一种物体出现在不同位置 → 特征相似
		            - 尺度已知：每个grid的物理尺寸固定（如0.4m×0.4m）
		        这些性质决定了：任何模型都应该利用"空间越近→关系越强"的先验
		    （2）CNN为什么天然匹配BEV
		        ① 局部感受野（3x3/5x5 conv）：只看邻域，=强加"空间越近→相关性越强"的约束
		        ② 平移不变性：同一卷积核在所有位置共享权重 → "车在左还是右"对模型无区别
		        ③ 层次化感受野：浅层看细节、深层看全局 → 自然地逐步扩大感受野
		        ④ 结果：更少参数 + 更容易收敛 + 更少数据需求 + 推理更稳定
		        一句话：CNN在做的是"世界是局部连续的"这份强结构假设，而BEV恰好满足这个假设
		    （3）Transformer在BEV上的四个问题
		        问题1——没有空间归纳偏置：
		            self-attention初始时任意位置↔任意位置权重相近
		            → 近距离点=远距离点（初始）→ 模型需要自己学会"什么是局部"→ 很难
		        问题2——计算复杂度爆炸：
		            BEV: 200×200=40000个grid
		            全局self-attention: O(N²)=1.6B → 无法直接计算
		            必须用降采样/稀疏attention/deformable attention → 引入近似误差
		        问题3——缺少连续性约束：
		            CNN天然平滑（卷积是滑动窗口平均）→ 相邻位置特征接近
		            Transformer的attention权重可以很"跳"→ 特征不连续/抖动
		            → 在BEV上表现为bbox抖动、检测不稳定
		        问题4——训练更难：
		            Transformer需要更多数据+更长训练+更强正则化
		            BEV监督本来就弱（尤其纯视觉），双重困难叠加
		    （4）直观对比
		        CNN在BEV上：
		            只看附近 → 慢慢扩散感受野 → 稳定建模空间结构
		        Transformer：
		            一上来全局连接 → 信息太多、信号太弱 → 学习困难
		        类比：CNN=在陌生城市从自己身边开始探索，Transformer=直接看整个地球→找不到重点
		    （5）工程维度对比
		        维度              | CNN                               | Transformer
		        GPU kernel优化     | Conv2d在cuDNN/TensorRT高度优化      | attention是memory-bound，kernel复杂
		        延迟稳定性          | 固定计算图，延迟可预测               | 动态性导致延迟抖动
		        显存              | 较低，中间tensor密集                 | 较高，需存attention矩阵
		        TensorRT部署       | 友好，经典算子fusion成熟             | 需要更多自定义优化
		    （6）BEVFormer为什么还能work（取了折中）
		        Deformable Attention：
		            不是全局attention → 每个query只采样少量关键点（如4~8点）
		            → 计算量从O(N²)降到O(N×points) → 可行且稳定
		        BEV Query设计：
		            每个query自带空间含义（绑定到某个(x,y)位置）
		            → 给query"空间锚点"→ 相当于给Transformer加"位置先验"
		        本质：BEVFormer=给Transformer加了一点CNN的bias（空间先验+局部采样）
		    （7）BEVFusion为什么选CNN做decoder
		        BEVFusion的decoder是2D CNN backbone+neck，不是Transformer decoder
		        原因：融合后的BEV特征就是一个规则网格 → CNN天然适配 → 稳定高效
		        而TransFusionHead虽然用了query，但只在200个proposal上做小Decoder，且query来自heatmap初始化（有空间先验），不是纯随机可学习的
		    （8）CNN vs Transformer 在BEV上的最终结论
		        不是一个比另一个"好"，而是：
		            - 规则网格上的特征精炼（backbone/neck）：CNN天然优势，稳定高效
		            - 物体级别的推理（检测head）：Transformer的query机制可以替代anchor/NMS，但需要加空间先验（位置编码/deformable attention）
		            - 全局上下文建模：Transformer有优势，但需要通过CNN先做局部精炼来降低搜索空间
		        最佳实践的组合思路：CNN负责局部特征精炼+Transformer负责物体级推理
		    （9）一句话总结
		        BEV是规则物理空间，空间连续性是其核心属性，CNN的局部卷积归纳偏置与此天然匹配→稳定快速；Transformer缺乏这种"空间先验"，需要从数据中自学习空间结构→困难且不稳定。BEVFormer/TransFusionHead能work正是因为通过Deformable Attention和位置编码给Transformer加了"空间约束"，本质上是CNN+Transformer的折中方案。
