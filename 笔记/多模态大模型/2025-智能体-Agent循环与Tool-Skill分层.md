> **归档来源**：`老笔记/暂时未归纳点.md` 第 147–239 行（按主题拆分归档）

---

	6、
	所属：智能体 Agent架构 Agent循环（Agent Loop）与 Tool / Skill 分层
	描述：Agent 会在每一步中从上下文和长期记忆中提取相关信息，结合工具描述和技能描述输入给大模型；大模型进行规划与决策，选择调用 Tool（原子操作）还是 Skill（复合能力）；执行结果后，将关键信息更新到记忆中，再进入下一步循环。核心流程分 6 步——① Observe（观察）：获取当前输入、上下文、检索相关记忆（RAG）；② Think / Plan（思考/规划）：判断是否需要拆任务、决定用 Tool 还是 Skill；③ Act（行动）：选择调用 Tool（原子操作，如搜索API）或 Skill（复合能力，如"深度研究"——Skill内部可再跑一个小Agent流程）或直接输出文本；④ Execute（执行）：Tool 直接执行返回结果，Skill 内部运行子流程（可能含多轮Tool调用+推理）；⑤ Reflect（反思/更新）：判断结果是否有用、是否需要继续或换策略；⑥ Update Memory（更新记忆）：写入关键结果和用户偏好（不是什么都存），然后进入下一轮循环。更深层理解：Agent 不是"一次性回答"的系统，而是一个持续运行的"感知→思考→行动→反思"循环体；Skill 的存在让 Agent 不必每步都从零开始——复杂套路被封装为可复用的能力模块，Agent 只需在合适的时机调用它们（Tool是"能做什么"，Skill是"怎么把事情做好"）。核心架构：Agent（总控）→ Skills（技能层，封装复合能力）→ Tools（工具层，原子操作）。
	冲突：
	    ① 容易误以为规划（Planning）是独立的一步——实际上有三种情况：隐式规划（最常见，ReAct模式，直接让模型一步步想，没有单独 Planner）、显式规划（高级 Agent，先生成完整计划再逐步执行）、动态规划（最强，每一步都重新规划，类似人类边做边想）
	    ② 容易误以为 Agent 的"记忆"就是对话历史——实际上分两种：短期记忆（当前上下文窗口内的对话和工具结果）和长期记忆（持久化的关键结果、用户偏好、经验教训），长期记忆需要通过 RAG 检索后注入上下文
	    ③ 容易误以为工具调用结果是"最终输出"——实际上工具结果是中间产物，Agent 需要 Reflect（反思）判断结果是否有用、是否继续调用工具
	    ④ 容易误以为 Agent Loop 每轮都一样——实际上随着上下文增长和记忆更新，每轮可用的信息不同，决策质量会逐步提升（或退化，如果上下文被污染）
	    ⑤ 容易误以为"更新记忆"就是全量存储——实际上是有选择地存：关键结果、用户偏好、重要经验，而不是什么都存（会引入噪声）
		    ⑥ 容易误以为 Skill 就是大号 Tool——实际上 Tool 是原子操作（无流程、无智能），Skill 是复合能力模块（包含 Tool + 流程 + LLM推理），Skill 内部可以有 loop/memory/planning
		    ⑦ 容易误以为 Agent 只能调用 Tool——实际上 Agent 可以调用 Skill（封装好的能力模块），Skill 让 Agent 不必每步都从零规划，极大提升稳定性和效率
		    ⑧ 容易误以为 Skill 内部是黑盒——实际上 Skill 有三种实现方式（Prompt模板 / Code编排 / 嵌套Agent），理解和区分这三种方式决定了你在什么场景选什么方案
	解决方法：
	    （1）Agent Loop 六步详解
	        Step 1 - Observe（观察）：获取当前用户输入 + 当前上下文窗口内容 + 从长期记忆中检索相关信息（RAG检索）
	        Step 2 - Think / Plan（思考/规划）：分析当前状态 → 判断任务复杂度 → 决定是否需要拆解子任务 → 确定下一步动作
	        Step 3 - Act（行动）：根据规划结果选择——调用 Tool（原子操作：函数调用 / API请求 / 代码执行），或调用 Skill（复合能力：Skill 内部编排多个 Tool + LLM推理），或直接输出文本回复
	        Step 4 - Execute（执行）：若选 Tool → 直接调用并获取结果；若选 Skill → Skill 内部启动子流程（可能是多轮 Tool 调用 + 推理循环），Skill 完成后返回汇总结果
	        Step 5 - Reflect（反思）：评估工具结果——是否解决了当前子问题？是否需要重试？是否需要换策略？结果是否足够回复用户？
	        Step 6 - Update Memory（记忆更新）：将关键信息写入长期记忆——用户偏好、本次任务的关键结果、经验教训；非关键信息不存
	    （2）规划的三种模式
	        隐式规划（ReAct）：最常见，模型边想边做，reasoning 和 action 交替进行，无独立 Planner 模块
	        显式规划：先生成完整执行计划（Plan-and-Solve），再逐步执行，适合复杂多步任务
	        动态规划：每一步都重新评估当前状态并调整计划，类似人类"边做边想边改"，最灵活但成本最高
	    （3）记忆系统设计
	        短期记忆 = 当前对话上下文窗口（有限的 token 容量，旧的被截断或遗忘）
	        长期记忆 = 持久化存储（数据库/向量库）→ 通过 RAG 检索相关内容注入当前上下文
	        记忆更新策略：不是全量存储 → 只存关键结果、用户偏好、重要经验 → 避免噪声污染
	    （4）Agent Loop 与传统程序的区别
	        传统程序：输入 → 固定逻辑处理 → 输出，流程确定
	        Agent Loop：输入 → 动态决策（调用 Tool 或 Skill，0~N次）→ 反思 → 更新状态 → 输出，流程动态
	        关键差异：Agent 每步都在"决策"，Tool/Skill 调用是"手段"而非"目的"，最终目标是完成任务而非执行固定流程；Skill 层的本质是"把已验证的套路沉淀为可复用模块"，让 Agent 在更高抽象层做决策
	    （5）Agent Loop 的工程实现要点
	        上下文管理：控制上下文长度，防止过长导致模型退化（lost-in-the-middle）
	        工具/Skill描述质量：Tool 描述和 Skill 描述是模型决策的依据 → 描述不清会导致错误调用；Skill 描述需说清楚"能做什么、内部做什么、何时该用何时不该用"
	        终止条件：设定最大步数 / 超时 / 任务完成判断，防止无限循环
	        错误处理：工具调用失败时的重试策略、降级策略、兜底策略
		        Skill 注册与发现：维护 Skill 目录（名称、描述、适用场景），让 Agent 在 Think 阶段能识别"该用哪个 Skill"
		        Skill 粒度设计：太粗 → 复用性差（只能在一个场景用）；太细 → 退化回 Tool，失去 Skill 的意义；合适的粒度 = 一个可独立完成的有意义子任务
	    （6）Skill 的定义与核心认知
	        Skill = 被封装好的"可复用能力模块"，比 Tool 更高级一层
	        一句话：Tool 是"能做什么"，Skill 是"怎么把事情做好"
	        Skill 本质上是"一个可以被调用的小Agent / 小流程"——内部包含 Tool + 流程 + 智能
	    （7）Tool vs Skill 核心对比
	        维度        | Tool（工具）        | Skill（技能）
	        粒度        | 很细（原子操作）    | 更粗（复合能力）
	        功能        | 单一操作           | 一组操作组合
	        是否需要LLM  | 不一定             | 通常需要
	        是否包含流程 | ❌                 | ✅
	        例子        | 搜索API、计算器     | 写报告、订旅行、深度研究
	        直观类比：Tool = search_flights() / book_hotel() / get_weather()（单一功能）
	                 Skill = plan_trip() → 内部调用 flights + weather + hotel + 组合逻辑（Tool + 流程 + 智能）
	    （8）带 Skill 的 Agent Loop（升级版）
	        1️⃣ Observe → 输入 + 上下文 + 记忆
	        2️⃣ Think / Plan → 决定：用 Tool？用 Skill？还是拆任务？
	        3️⃣ Act（关键变化）→ 选择：调用 Tool（原子操作）或调用 Skill（复合能力）
	        4️⃣ Execute → Tool直接执行，或 Skill 内部再跑一个"小Agent流程"（嵌套Agent）
	        5️⃣ Reflect → 评估结果
	        6️⃣ Update Memory → 更新关键信息
	        关键认知：Skill 把"复杂套路"固化到可复用的模块里，Agent 不需要每步都重新想——提高稳定性、提高效率、更像人类经验
	    （9）Skill 的三种实现方式
	        ① Prompt Skill（最简单）：本质是一个 Prompt 模板
	            例："你是一个写邮件专家，请帮我写一封正式邮件..."
	            特点：纯文本指令，无代码逻辑，最轻量
	        ② Code Skill（工程常见）：LLM + 工具组合
	            例：def write_report(data): summary = llm("总结数据"); chart = draw_chart(data); return summary + chart
	            特点：用代码编排 LLM 调用和工具调用，有确定性流程
	        ③ Agent Skill（高级🔥）：内部有 loop + memory + planning
	            例：def research_topic(topic): while not done: search(); summarize(); refine()
	            特点：Agent inside Agent（嵌套智能体），自己也是一个完整的感知→思考→行动循环
	    （10）为什么需要 Skill（没有 Skill 的世界的问题）
	        ❌ 只有 Tool 时：复杂任务每一步都要重新想 → 不稳定、容易走偏、效率低
	        ✅ 有 Skill 后：把"复杂套路"固化为可复用模块 → 稳定、高效、可组合
	        类比：Tool = 手里的工具（锤子），Skill = 能力（会修桌子）——锤子本身不会修桌子，但"修桌子"这个 Skill 知道什么时候用锤子、怎么用
	    （11）Agent 分层架构（最终抽象）
	        Agent（总控）
	        ├── Memory（记忆）
	        ├── Planner（规划）
	        ├── Skills（技能）👈 核心抽象层
	        │     ├── Skill A（写作）
	        │     ├── Skill B（搜索分析）
	        │     └── Skill C（数据处理）
	        └── Tools（工具）
	              ├── API
	              ├── DB
	              └── Code
	        核心关系：Tool → 被 Skill 使用；Skill → 被 Agent 调用
	        没有 Skill 层 → Agent 直接操作 Tool → 每步都要重新决策，不稳定
	        有 Skill 层 → Agent 调用封装好的能力模块 → 稳定、高效、可复用
	    （12）Skill 与相关概念的区分
	        Skill vs Workflow：Workflow 是"多个步骤的编排顺序"（DAG），Skill 是"封装好的能力模块"——Workflow 关注"怎么做"，Skill 关注"能做什么"
	        Skill vs Function Calling：Function Calling 是 LLM 调用单个函数的机制（属于 Tool 层），Skill 是更高层的封装——一个 Skill 内部可能调用多次 Function Calling
	        Skill vs Chain（LangChain）：Chain 是固定的管道（pipeline），Skill 更灵活——可以有内部循环、条件分支、动态决策
