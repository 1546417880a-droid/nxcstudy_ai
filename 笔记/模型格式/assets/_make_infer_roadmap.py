# -*- coding: utf-8 -*-
"""生成「推理加速技术路线」四张图。

图 1  三堵墙：所有加速技术都在翻其中一堵
图 2  路线全景：同一份训练产物，五条部署路线
图 3  ONNX → TensorRT：图纸与机器
图 4  精度阶梯：FP32 → INT4

用法：python3 _make_infer_roadmap.py <输出目录>
"""
import os
import sys

from _lay import (
    Lay, INK, MUTED, FAINT, LINE, HAIR, BG, WHITE, SLATE,
    BLUE, BLUE_T, BLUE_L, GREEN, GREEN_T, GREEN_L,
    ORANGE, ORANGE_T, ORANGE_L, TEAL, TEAL_T, TEAL_L,
)

VIOLET = (109, 40, 217)
VIOLET_T = (245, 243, 255)
VIOLET_L = (221, 214, 254)
ROSE = (190, 18, 60)
ROSE_T = (255, 241, 244)
ROSE_L = (253, 205, 215)
GREY = (100, 116, 139)
GREY_T = (248, 250, 252)


def flow(L, x, y, items, h=46, sz=16, gap=28, muted=FAINT):
    """横向芯片流：items = [(text, fg, bg, border), ...]，末项通常是高亮产物。"""
    cx = x
    for i, (t, fg, bg, bd) in enumerate(items):
        w = L.chip(cx, y, t, sz, fg, bg, bd, 17, h, True, h / 2)
        if i < len(items) - 1:
            L.arrow(cx + w + 7, y + h / 2, cx + w + gap - 5, y + h / 2, (203, 213, 225), 1.8, 8)
        cx += w + gap
    return cx


def pill(L, s):
    w = L.tw(s, 17, True) + 34
    L.chip(2344 - w, 44, s, 17, SLATE, (241, 245, 249), LINE, 17, 40, True)


# ======================================================================
# 图 1  路线全景
# ======================================================================
ROUTES = [
    dict(k="A", name="保持 PyTorch", tag="不换格式，只换执行方式", accent=GREY,
         tint=GREY_T, light=(203, 213, 225),
         items=[(".pt", GREY, WHITE, LINE),
                ("torch.compile", INK, WHITE, LINE),
                ("Triton / C++ kernel", INK, WHITE, LINE),
                ("进程内运行", SLATE, GREY_T, (203, 213, 225))],
         scene="研究 / 调试", scene2="形状多变",
         note="编译期：Inductor 把算子融合并生成 kernel，产物不落盘，只在进程内生效",
         note2="代价：首次编译要等；动态形状会触发重编译（recompile）"),

    dict(k="B", name="换通用格式", tag="一次导出，多硬件运行", accent=BLUE,
         tint=BLUE_T, light=BLUE_L,
         items=[(".pt", GREY, WHITE, LINE),
                ("ONNX", BLUE, BLUE_T, BLUE_L),
                ("ONNX Runtime", INK, WHITE, LINE),
                ("执行提供器 EP", BLUE, BLUE_T, BLUE_L)],
         scene="CPU / x86", scene2="跨平台",
         note="编译期：加载图时做局部融合与算子选择，优化偏保守，胜在通用",
         note2="换硬件只需换 EP：CUDA / TensorRT / OpenVINO / DirectML 都是插件"),

    dict(k="C", name="换专用引擎", tag="为这块 GPU 编译一次", accent=TEAL,
         tint=TEAL_T, light=TEAL_L,
         items=[(".pt", GREY, WHITE, LINE),
                ("ONNX", BLUE, BLUE_T, BLUE_L),
                ("TensorRT 编译", INK, WHITE, LINE),
                ("engine .plan", TEAL, TEAL_T, TEAL_L)],
         scene="NVIDIA GPU", scene2="极致性能",
         note="编译期：全图融合 + kernel 自动调优 + 内存规划 + 精度选择，优化最激进",
         note2="代价：engine 绑死 GPU 型号与 TensorRT 版本，换卡要重编"),

    dict(k="D", name="换硬件生态", tag="为端侧芯片换一套工具链", accent=ORANGE,
         tint=ORANGE_T, light=ORANGE_L,
         items=[(".pt", GREY, WHITE, LINE),
                ("ONNX", BLUE, BLUE_T, BLUE_L),
                ("OpenVINO / CoreML", INK, WHITE, LINE),
                ("TFLite / NCNN / RKNN", ORANGE, ORANGE_T, ORANGE_L)],
         scene="端侧 / 嵌入式", scene2="手机·车机·板卡",
         note="编译期：各家自带优化器，针对自家硬件指令集与内存层级做定制",
         note2="代价：算子覆盖不全，模型改动常要回头改结构"),

    dict(k="E", name="大模型专用", tag="不是编译模型，是管请求", accent=VIOLET,
         tint=VIOLET_T, light=VIOLET_L,
         items=[(".pt", GREY, WHITE, LINE),
                ("vLLM / TensorRT-LLM", INK, WHITE, LINE),
                ("SGLang / TGI", INK, WHITE, LINE),
                ("服务化引擎", VIOLET, VIOLET_T, VIOLET_L)],
         scene="LLM 在线服务", scene2="高并发",
         note="编译期：算子融合 + KV cache 分页管理 + 连续批处理，重点在吞吐不是单条延迟",
         note2="和 A–D 不是替代关系：它内部同样会用到 ONNX / TensorRT 那一套"),
]


def fig1(path):
    W, H = 2400, 1390
    L = Lay(W, H)

    L.text(56, 36, "从训练完的模型，到跑得快：五条路线", 42, INK, "la", True)
    L.text(56, 100,
           "同一份 .pt 可以走五条完全不同的路。差别不在「格式」，而在「谁替你在编译期做优化、优化得多狠」",
           21, MUTED)
    pill(L, "图 2 / 6 · 路线全景")

    # ---------------- 起点 ----------------
    SX0, SX1 = 56, 300
    SY0, SY1 = 250, 706
    L.box(SX0, SY0, SX1, SY1, 18, INK, INK, 0)
    L.text(SX0 + 24, SY0 + 22, "训练产物", 23, WHITE, "la", True)
    L.text(SX0 + 24, SY0 + 56, "三样缺一不可", 13.5, (148, 163, 184))

    for i, (t1, t2) in enumerate([("model.py", "定义网络结构"),
                                 ("state_dict.pt", "只有权重，没有结构"),
                                 ("model.eval()", "关掉 BN / Dropout")]):
        y = SY0 + 86 + i * 76
        L.box(SX0 + 18, y, SX1 - 18, y + 62, 10, (30, 41, 59), (45, 58, 78), 1.2)
        L.text(SX0 + 32, y + 12, t1, 14.5, WHITE, "la", True)
        L.text(SX0 + 32, y + 35, t2, 12, (148, 163, 184), maxw=200)

    # 左侧补充说明，平衡构图
    L.text(SX0 + 2, SY1 + 34, "五条路在翻同样的三堵墙", 15.5, INK, "la", True)
    for j, s_ in enumerate(["少发几次指令", "少在显存里往返", "少用几位精度"]):
        L.d.ellipse([(SX0 + 4) * 2, (SY1 + 66 + j * 30 + 7) * 2,
                     (SX0 + 10) * 2, (SY1 + 66 + j * 30 + 13) * 2], fill=TEAL)
        L.text(SX0 + 20, SY1 + 66 + j * 30, s_, 14, SLATE)
    L.text(SX0 + 2, SY1 + 178, "详见 图 2", 12.5, FAINT, "la", True)

    # ---------------- 五条路线 ----------------
    RX0, RX1 = 360, 2344
    RY0, RH, RGAP = 250, 200, 14
    TRUNK = 330

    # 主干
    L.line([(TRUNK, RY0 + RH / 2), (TRUNK, RY0 + 4 * (RH + RGAP) + RH / 2), ], (203, 213, 225), 2.2)
    L.line([(SX1 + 2, RY0 + RH / 2), (TRUNK, RY0 + RH / 2)], (203, 213, 225), 2.2)

    for i, r in enumerate(ROUTES):
        y = RY0 + i * (RH + RGAP)
        a, t, li = r["accent"], r["tint"], r["light"]

        L.box(RX0, y, RX1, y + RH, 16, WHITE, LINE, 1.6)
        L.line([(RX0 + 4, y + 18), (RX0 + 4, y + RH - 18)], a, 5)

        # 左标签
        L.text(RX0 + 26, y + 22, r["k"], 38, a, "la", True)
        L.text(RX0 + 76, y + 28, r["name"], 21, INK, "la", True, maxw=170)
        L.text(RX0 + 76, y + 60, r["tag"], 13.5, MUTED, maxw=176)
        L.line([(RX0 + 264, y + 22), (RX0 + 264, y + RH - 22)], HAIR, 1.4)

        # 芯片流
        flow(L, RX0 + 286, y + 40, r["items"])

        # 右上场景
        tag = f'{r["scene"]} · {r["scene2"]}'
        tw = L.tw(tag, 13, True) + 28
        L.box(RX1 - 24 - tw, y + 24, RX1 - 24, y + 24 + 32, 16, t, a, 1.2)
        L.text(RX1 - 24 - tw / 2, y + 24 + 16, tag, 13, a, "mm", True)

        # 说明
        L.text(RX0 + 286, y + 112, r["note"], 15.5, SLATE, maxw=RX1 - RX0 - 320)
        L.text(RX0 + 286, y + 144, r["note2"], 14.5, MUTED, maxw=RX1 - RX0 - 320)

        # 从总线引出
        cy = y + RH / 2
        L.line([(TRUNK, cy), (RX0 - 12, cy)], (203, 213, 225), 2.2)
        L.arrow(RX0 - 22, cy, RX0 - 4, cy, (203, 213, 225), 2.2, 10)

    # ---------------- 结论条 ----------------
    by = 1344
    L.text(56, by, "一句话：A 不换格式；B 换通用格式；C 换专用引擎；D 换硬件生态；E 换服务形态。"
                   "只有 C 和 E 是「编译期把性能榨干」，B 和 D 是「用通用性换省事」", 16.5, SLATE)
    L.text(2344, by, "nxcstudy_ai · 模型格式 · 2026-11", 15, FAINT, "ra")

    L.save(path)


# ======================================================================
# 图 2  三堵墙
# ======================================================================
WALLS = [
    dict(n="①", name="CPU 发号施令", en="launch overhead", accent=BLUE, tint=BLUE_T, light=BLUE_L,
         ess="GPU 算得比 CPU 派活快。小 kernel 一个接一个排队，GPU 一直在等下一道命令。",
         sym=["batch 很小、形状很多", "GPU 利用率上不去（< 50%）", "算子和算子之间有空隙"],
         tools=["CUDA Graph", "torch.compile", "算子融合", "静态图", "增大 batch"],
         eg="ResNet50 单图、batch=1 时，GPU 利用率常常只有 20–30%，时间全花在等 Python 发 kernel",
         meas="nsys 看时间轴：kernel 之间的空白就是浪费"),
    dict(n="②", name="显存带宽", en="memory bound", accent=TEAL, tint=TEAL_T, light=TEAL_L,
         ess="中间结果反复写回显存再读出来。算力闲着，带宽先打满 —— 这是最常见的瓶颈。",
         sym=["算力利用率低，但带宽接近峰值", "融合之后立刻变快", "大模型比小模型更明显"],
         tools=["fusion 不落地", "内存复用", "FP16 / BF16", "INT8 / FP8", "PagedAttention"],
         eg="Conv+BN+ReLU 中间要写两次显存；融合后只写一次，这部分带宽直接省三分之二",
         meas="nsight compute 看 DRAM throughput 是否顶到峰值"),
    dict(n="③", name="算力 / 精度", en="compute bound", accent=ORANGE, tint=ORANGE_T, light=ORANGE_L,
         ess="真的算不动了。FLOPs 就那么多，只能换更强的硬件或更低的精度。",
         sym=["大 batch 时耗时线性增长", "算力利用率 > 80%", "再怎么融合也快不了多少"],
         tools=["Tensor Core", "INT8 / FP8", "kernel 自动调优", "张量并行", "稀疏 / 蒸馏"],
         eg="batch 加到某个点后耗时随 batch 线性增长 —— 这时只能堆显卡或降精度，融合已经没用了",
         meas="看 SM 占用率与 TFLOPS 是否接近理论上限"),
]


def fig2(path):
    W, H = 2400, 1032
    L = Lay(W, H)

    L.text(56, 36, "为什么推理慢：三堵墙，所有加速技术都在翻其中一堵", 42, INK, "la", True)
    L.text(56, 100,
           "先判断撞的是哪堵墙，再决定用哪个工具 —— 顺序反了会白忙：给带宽瓶颈上 kernel 调优，等于给堵车路段换发动机",
           21, MUTED)
    pill(L, "图 1 / 6 · 三堵墙")

    CW, GAP = 736, 40
    Y0, Y1 = 178, 866

    for i, w in enumerate(WALLS):
        x0 = 56 + i * (CW + GAP)
        x1 = x0 + CW
        a, t, li = w["accent"], w["tint"], w["light"]

        L.box(x0, Y0, x1, Y1, 18, WHITE, LINE, 1.6)

        # 头
        L.box(x0 + 20, Y0 + 20, x1 - 20, Y0 + 96, 14, t, t, 0)
        L.text(x0 + 40, Y0 + 38, w["n"], 30, a, "la", True)
        L.text(x0 + 82, Y0 + 34, w["name"], 26, INK, "la", True, maxw=CW - 120)
        L.text(x0 + 82, Y0 + 68, w["en"], 14, a, "la", True)

        # 本质
        L.text(x0 + 24, Y0 + 116, "本质", 14, a, "la", True)
        L.text(x0 + 24, Y0 + 142, w["ess"], 15.5, SLATE, maxw=CW - 48)

        # 症状
        sy = Y0 + 242
        L.line([(x0 + 22, sy - 18), (x1 - 22, sy - 18)], HAIR, 1.4)
        L.text(x0 + 24, sy, "怎么判断你撞的是它", 14, MUTED, "la", True)
        for j, s in enumerate(w["sym"]):
            yy = sy + 34 + j * 30
            L.d("ellipse",) if False else None
            L.d.ellipse([(x0 + 27) * 2, (yy + 8) * 2, (x0 + 33) * 2, (yy + 14) * 2], fill=a)
            L.text(x0 + 44, yy, s, 15, SLATE, maxw=CW - 76)

        # 工具
        ty = sy + 34 + len(w["sym"]) * 30 + 22
        L.line([(x0 + 22, ty - 16), (x1 - 22, ty - 16)], HAIR, 1.4)
        L.text(x0 + 24, ty, "翻墙的工具", 14, MUTED, "la", True)

        cx, cy = x0 + 24, ty + 30
        for tool in w["tools"]:
            tw = L.tw(tool, 14.5, True) + 30
            if cx + tw > x1 - 24:
                cx = x0 + 24
                cy += 44
            L.box(cx, cy, cx + tw, cy + 36, 18, t, li, 1.2)
            L.text(cx + tw / 2, cy + 18, tool, 14.5, a, "mm", True)
            cx += tw + 12

        # 举例
        ey = cy + 44 + 34
        L.line([(x0 + 22, ey - 16), (x1 - 22, ey - 16)], HAIR, 1.4)
        L.text(x0 + 24, ey, "举个例", 14, MUTED, "la", True)
        L.text(x0 + 24, ey + 26, w["eg"], 14.5, SLATE, maxw=CW - 48)

        # 怎么量
        my = Y1 - 92
        L.box(x0 + 20, my, x1 - 20, Y1 - 20, 12, (248, 250, 252), LINE, 1.2)
        L.text(x0 + 34, my + 14, "怎么量", 12.5, FAINT, "la", True)
        L.text(x0 + 34, my + 38, w["meas"], 14, SLATE, maxw=CW - 68)

    # 底部顺序条
    by = Y1 + 28
    L.box(56, by, 2344, by + 106, 14, INK, INK, 0)
    L.text(88, by + 16, "排查顺序", 16, (148, 163, 184), "la", True)
    steps = [("先看 GPU 利用率", "低 → 撞的是 ①"), ("再量 DRAM 带宽", "满 → 撞的是 ②"),
             ("最后看算力", "满 → 撞的是 ③"), ("只有 ③ 才值得谈低精度与并行", "")]
    sx = 240
    for i, (s1, s2) in enumerate(steps):
        L.text(sx, by + 18, s1, 17, WHITE, "la", True)
        if s2:
            L.text(sx, by + 46, s2, 14, TEAL_L, "la")
        if i < 3:
            L.arrow(sx + L.tw(s1, 17, True) + 18, by + 30, sx + L.tw(s1, 17, True) + 52, by + 30,
                    (71, 85, 105), 2.2, 10)
        sx += max(L.tw(s1, 17, True), L.tw(s2, 14)) + 96

    L.save(path)


# ======================================================================
# 图 3  ONNX → TensorRT
# ======================================================================
def fig3(path):
    W, H = 2400, 1240
    L = Lay(W, H)

    L.text(56, 36, "ONNX 是一张图纸，TensorRT 是按这张图纸造的机器", 42, INK, "la", True)
    L.text(56, 100,
           "一个负责「说清楚要算什么」，一个负责「在你的这块 GPU 上算到最快」—— 它们不是竞品，是上下游",
           21, MUTED)
    pill(L, "图 4 / 6 · 图纸与机器")

    # ---------------- 四段流水 ----------------
    boxes = [
        ("① PyTorch 模型", "动态图，每次 forward 现建图", GREY, GREY_T, (203, 213, 225)),
        ("② model.onnx", "静态计算图，只描述语义", BLUE, BLUE_T, BLUE_L),
        ("③ model.plan", "GPU 专用执行计划", TEAL, TEAL_T, TEAL_L),
        ("④ 推理结果", "加载 → 分配 → 执行 → 输出", VIOLET, VIOLET_T, VIOLET_L),
    ]
    labels = ["torch.onnx.export", "trtexec / TensorRT API", "TensorRT Runtime"]
    BW, GAP = 470, 130
    y0, y1 = 200, 330
    for i, (t1, t2, c, tc, lc) in enumerate(boxes):
        x0 = 56 + i * (BW + GAP)
        x1 = x0 + BW
        L.box(x0, y0, x1, y1, 16, tc, lc, 1.6)
        L.text(x0 + 24, y0 + 26, t1, 22, c, "la", True, maxw=BW - 48)
        L.text(x0 + 24, y0 + 66, t2, 14.5, MUTED, maxw=BW - 48)
        if i < 3:
            ax0, ax1 = x1 + 12, x1 + GAP - 12
            L.arrow(ax0, (y0 + y1) / 2, ax1, (y0 + y1) / 2, (148, 163, 184), 2.4, 12)
            L.text((ax0 + ax1) / 2, y0 - 26, labels[i], 13, MUTED, "mm", True)

    # 编译期 / 运行期 分界
    L.dash(56, 372, 2344, 372, (203, 213, 225), 1.6, 8, 7)
    L.text(56, 352, "编译期 · 只做一次", 13.5, TEAL, "la", True)
    L.text(2344, 352, "运行期 · 每帧都做", 13.5, VIOLET, "ra", True)

    # ---------------- 三个说明列 ----------------
    cols = [
        dict(x0=56, x1=760, title="导出 ONNX 时做了什么", accent=BLUE,
             rows=[("做了", ["trace 一次真实 forward，记下走过的分支",
                             "PyTorch 算子 1:1 映射成 ONNX 算子",
                             "固化成静态图，落盘成 .onnx"]),
                   ("没做", ["不做算子融合", "不做 kernel 选择", "不做内存规划"])],
             foot="重点在语义正确，不在性能",
             extra=("导出时最容易踩的三个坑",
                    ["动态 shape 没写 dynamic_axes，导出后只认固定尺寸",
                     "用了 PyTorch 特有算子，ONNX 找不到对应，直接导不出",
                     "导出后不做数值比对，等到精度掉下来才发现算子语义不同"])),
        dict(x0=792, x1=1720, title="TensorRT 编译的五步", accent=TEAL,
             rows=[("①", ["解析 ONNX 图，建立内部表示"]),
                   ("②", ["图优化：常量折叠、消除冗余、垂直/水平融合"]),
                   ("③", ["kernel 自动调优：试多种实现，实测选最快"]),
                   ("④", ["内存规划：分析张量生命周期，复用不重叠的显存"]),
                   ("⑤", ["精度选择 FP16 / INT8 + 序列化成 .plan"])],
             foot="编译期决定了性能上限，运行期只是照着跑"),
        dict(x0=1752, x1=2344, title="运行期做什么", accent=VIOLET,
             rows=[("1", ["反序列化 engine，创建 execution context"]),
                   ("2", ["分配输入输出 buffer（或用统一显存）"]),
                   ("3", ["H2D 拷贝 → enqueue 到 CUDA stream"]),
                   ("4", ["等 stream 完成，D2H 取结果"])],
             foot="CPU 几乎不参与，只发一次「开始执行」",
             extra=("engine / context / stream 三件东西",
                    ["engine：编译好的程序，可落盘复用，一模型一份",
                     "context：运行实例，一个 engine 可开多个，共享权重",
                     "stream：执行队列，多 context 可并行跑在不同 stream"])),
    ]

    for c in cols:
        x0, x1, a = c["x0"], c["x1"], c["accent"]
        L.box(x0, 400, x1, 1040, 16, WHITE, LINE, 1.6)
        L.text(x0 + 22, 418, c["title"], 18, a, "la", True, maxw=x1 - x0 - 44)
        L.line([(x0 + 20, 452), (x1 - 20, 452)], HAIR, 1.4)

        cy = 470
        for tag, items in c["rows"]:
            good = tag in ("做了", "没做") or tag.isdigit()
            tc = GREEN if tag == "做了" else (ROSE if tag == "没做" else a)
            bg = GREEN_T if tag == "做了" else (ROSE_T if tag == "没做" else BLUE_T)
            L.box(x0 + 20, cy, x0 + 20 + L.tw(tag, 13, True) + 22, cy + 24, 12, bg, bg, 0)
            L.text(x0 + 31, cy + 12, tag, 13, tc, "lm", True)
            cy += 38
            for it in items:
                L.text(x0 + 26, cy, "·", 15, (203, 213, 225))
                L.text(x0 + 40, cy, it, 14.5, SLATE, maxw=x1 - x0 - 70)
                cy += 32 if L.tw(it, 14.5) < (x1 - x0 - 80) else 54
            cy += 6

        if c.get("extra"):
            t_, items_ = c["extra"]
            ey = cy + 24
            L.line([(x0 + 22, ey - 14), (x1 - 22, ey - 14)], HAIR, 1.4)
            L.text(x0 + 24, ey, t_, 14, MUTED, "la", True)
            for j, it in enumerate(items_):
                yy = ey + 30 + j * 34
                L.text(x0 + 24, yy, "·", 15, (203, 213, 225))
                L.text(x0 + 38, yy, it, 14, SLATE, maxw=x1 - x0 - 70)

        L.line([(x0 + 20, 980), (x1 - 20, 980)], HAIR, 1.4)
        L.text(x0 + 22, 994, c["foot"], 13.5, a, "la", True, maxw=x1 - x0 - 44)

    # ---------------- 隐喻条 ----------------
    by = 1070
    L.box(56, by, 2344, by + 130, 16, INK, INK, 0)
    L.text(88, by + 22, "为什么不能只记「TensorRT 快是因为融合」", 22, WHITE, "la", True)
    L.text(88, by + 64,
           "融合只是「少写几次显存」；真正拉开差距的是另外三件：kernel 自动调优（选对的实现）、"
           "内存复用（不重复申请）、精度优化（FP16 / INT8）—— 四件事一起做，才有量级差别", 16, (148, 163, 184))
    L.text(88, by + 96, "ONNX Runtime 也会融合，但它不做自动调优、全局内存规划也更保守，所以即使都融合仍有明显差距",
           15, TEAL_L)

    L.text(56, H - 34, "图 3 · 从 PyTorch 到 TensorRT：每步做了什么、没做什么", 17, FAINT)
    L.text(2344, H - 34, "nxcstudy_ai · 模型格式 · 2026-11", 17, FAINT, "ra")
    L.save(path)


# ======================================================================
# 图 4  精度阶梯
# ======================================================================
LEVELS = [
    dict(n="FP32", bits=32, mem=100, speed="1×", how="原样", risk="无", c=GREY, t=GREY_T,
         l=(203, 213, 225), note="训练默认。显存和带宽都最贵"),
    dict(n="FP16 / BF16", bits=16, mem=50, speed="2–8×", how="直接转，无需校准", risk="小", c=BLUE,
         t=BLUE_T, l=BLUE_L, note="有 Tensor Core 就免费提速；BF16 范围更大更好训"),
    dict(n="FP8", bits=8, mem=25, speed="4–16×", how="需要缩放因子", risk="中", c=TEAL,
         t=TEAL_T, l=TEAL_L, note="H100 之后的新档位，精度接近 INT8 但更好调"),
    dict(n="INT8", bits=8, mem=25, speed="4–16×", how="需要校准数据", risk="中", c=ORANGE,
         t=ORANGE_T, l=ORANGE_L, note="工业界最成熟的一档，图像/检测类几乎无损"),
    dict(n="INT4", bits=4, mem=13, speed="8–32×", how="必须 QAT 或分组量化", risk="大", c=ROSE,
         t=ROSE_T, l=ROSE_L, note="LLM 权重常用；激活一般仍留 8 bit"),
]


def fig4(path):
    W, H = 2400, 1040
    L = Lay(W, H)

    L.text(56, 36, "精度阶梯：FP32 → INT4，每降一档换到什么、失去什么", 42, INK, "la", True)
    L.text(56, 100,
           "显存和带宽近似按位宽比例下降，速度取决于有没有对应的 Tensor Core —— 但每一档的「可控性」在变差",
           21, MUTED)
    pill(L, "图 6 / 6 · 精度阶梯")

    CW, GAP = 440, 22
    Y0 = 200
    for i, v in enumerate(LEVELS):
        x0 = 56 + i * (CW + GAP)
        x1 = x0 + CW
        L.box(x0, Y0, x1, 704, 16, WHITE, LINE, 1.6)

        # 名称
        L.box(x0 + 18, Y0 + 18, x1 - 18, Y0 + 88, 12, v["t"], v["t"], 0)
        L.text(x0 + 34, Y0 + 38, v["n"], 24, v["c"], "la", True, maxw=CW - 60)
        L.text(x1 - 32, Y0 + 62, f'{v["bits"]} bit', 14, v["c"], "ra", True)

        # 显存条
        L.text(x0 + 22, Y0 + 106, "相对显存", 13, MUTED, "la", True)
        barw = (CW - 44) * v["mem"] / 100.0
        L.box(x0 + 22, Y0 + 126, x1 - 22, Y0 + 152, 6, (241, 245, 249), (241, 245, 249), 0)
        if barw > 2:
            L.box(x0 + 22, Y0 + 126, x0 + 22 + barw, Y0 + 152, 6, v["c"], v["c"], 0)
        L.text(x1 - 22, Y0 + 106, f'{v["mem"]}%', 14, v["c"], "ra", True)

        # 速度
        L.text(x0 + 22, Y0 + 176, "典型加速", 13, MUTED, "la", True)
        L.text(x0 + 22, Y0 + 200, v["speed"], 26, v["c"], "la", True)

        L.line([(x0 + 20, Y0 + 248), (x1 - 20, Y0 + 248)], HAIR, 1.4)

        # 怎么得到
        L.text(x0 + 22, Y0 + 262, "怎么得到", 13, MUTED, "la", True)
        L.text(x0 + 22, Y0 + 286, v["how"], 15, SLATE, maxw=CW - 44)

        # 风险
        L.text(x0 + 22, Y0 + 330, "精度风险", 13, MUTED, "la", True)
        L.text(x0 + 22, Y0 + 354, v["risk"], 15, v["c"], "la", True)

        L.line([(x0 + 20, Y0 + 392), (x1 - 20, Y0 + 392)], HAIR, 1.4)
        L.text(x0 + 22, Y0 + 406, v["note"], 14, SLATE, maxw=CW - 44)

        # 阶梯连线
        if i < len(LEVELS) - 1:
            L.arrow(x1 + 2, Y0 + 60, x1 + GAP - 2, Y0 + 60, (203, 213, 225), 2, 9)

    # ---------------- PTQ vs QAT ----------------
    by = 736
    L.text(56, by, "两条得到低精度的路", 20, INK, "la", True)

    cards = [
        ("PTQ · 训练后量化", BLUE, BLUE_T, BLUE_L,
         ["模型已经训好，只跑几十~几百张图统计每层的数值范围",
          "调几个超参就能出结果，不需要重新训练",
          "INT8 的常态做法；INT4 往往不够用"]),
        ("QAT · 量化感知训练", ORANGE, ORANGE_T, ORANGE_L,
         ["训练时就模拟量化误差，让权重学会「适应」低精度",
          "要重训，成本高，但 INT4 及以下基本必须走它",
          "常见做法：先 PTQ 拿到基线，再 QAT 微调"]),
    ]
    for i, (t1, c, tc, lc, items) in enumerate(cards):
        x0 = 56 + i * 1150
        x1 = x0 + 1112
        L.box(x0, by + 40, x1, by + 216, 14, tc, lc, 1.4)
        L.text(x0 + 22, by + 56, t1, 18, c, "la", True)
        for j, s in enumerate(items):
            L.text(x0 + 22, by + 92 + j * 34, "·", 15, c)
            L.text(x0 + 36, by + 92 + j * 34, s, 14, SLATE, maxw=x1 - x0 - 60)

    L.text(56, H - 64,
           "一句记住：显存和带宽按位宽线性下降，速度是台阶式跳变（因为有 Tensor Core 才算数），"
           "而精度风险是指数级上升 —— 越往下，越要靠数据验证而不是靠感觉", 16.5, SLATE)
    L.text(56, H - 26, "图 4 · 精度阶梯与 PTQ / QAT 两条路", 15, FAINT)
    L.text(2344, H - 26, "nxcstudy_ai · 模型格式 · 2026-11", 15, FAINT, "ra")
    L.save(path)


# ======================================================================
# 图 4  四招组合拳
# ======================================================================
def dots(L, x, y, n, color, r=6.5, gap=19):
    for i in range(3):
        cx = x + i * gap
        if i < n:
            L.d.ellipse([(cx - r) * 2, (y - r) * 2, (cx + r) * 2, (y + r) * 2], fill=color)
        else:
            L.d.ellipse([(cx - r) * 2, (y - r) * 2, (cx + r) * 2, (y + r) * 2],
                        fill=WHITE, outline=(226, 232, 240), width=3)


COMBO = [
    dict(name="融合", en="Kernel Fusion", accent=BLUE, tint=BLUE_T, light=BLUE_L,
         one="多个算子合成一个 kernel",
         cells=[(2, "少发几次命令"), (3, "只搬一次"), (0, "不改变 FLOPs"), (2, "中间结果不占坑")]),
    dict(name="自动调优", en="Auto-tuning", accent=TEAL, tint=TEAL_T, light=TEAL_L,
         one="实测选出最快的实现",
         cells=[(0, "编译期完成"), (1, "按卡按尺寸选"), (3, "同 FLOPs 更快"), (0, "不改变占用")]),
    dict(name="内存优化", en="Memory", accent=VIOLET, tint=VIOLET_T, light=VIOLET_L,
         one="复用不重叠的显存块",
         cells=[(1, "少申请少释放"), (2, "L2 命中率↑"), (0, "不改变算力"), (3, "峰值大幅下降")]),
    dict(name="精度优化", en="FP16 / INT8", accent=ORANGE, tint=ORANGE_T, light=ORANGE_L,
         one="用更少的位存同一个数",
         cells=[(1, "拷贝量变小"), (3, "数据量减半"), (3, "Tensor Core"), (3, "显存减半")]),
]


def fig_combo(path):
    W, H = 2400, 1290
    L = Lay(W, H)

    L.text(56, 36, "四招不是四个开关，是一套互相放大的组合拳", 42, INK, "la", True)
    L.text(56, 100,
           "每一招省的「东西」不一样：融合省搬运、调优省计算、内存省峰值、精度三样一起省 —— 而且它们会互相加成",
           21, MUTED)
    pill(L, "图 5 / 6 · 组合拳")

    # ---------------- 第一层：矩阵 ----------------
    L.text(56, 158, "第一层 · 每一招各自打在哪", 19, INK, "la", True)

    HY0, HY1 = 196, 240
    RX0, RX1 = 56, 2344
    C0, C1 = 470, 2344          # 三列起点
    COLW = (C1 - C0) / 4

    heads = [("① 发号施令", "launch", BLUE, BLUE_T),
             ("② 显存带宽", "bandwidth", TEAL, TEAL_T),
             ("③ 算力", "compute", ORANGE, ORANGE_T),
             ("④ 显存容量", "capacity", GREY, GREY_T)]

    L.box(RX0, HY0, C0 - 20, HY1, 10, (241, 245, 249), LINE, 1.2)
    L.text(RX0 + 20, (HY0 + HY1) / 2, "四招", 14, MUTED, "lm", True)
    for i, (h1, h2, c, t) in enumerate(heads):
        x0 = C0 + i * COLW
        L.box(x0, HY0, x0 + COLW - 8, HY1, 10, t, t, 0)
        L.text(x0 + COLW / 2 - 4, HY0 + 13, h1, 17, c, "ma", True)
        L.text(x0 + COLW / 2 - 4, HY0 + 34, h2, 12, MUTED, "ma")

    RY = HY1 + 8
    RH = 92
    for i, v in enumerate(COMBO):
        y = RY + i * RH
        a, t, li = v["accent"], v["tint"], v["light"]
        L.box(RX0, y, RX1, y + RH - 10, 12, WHITE, LINE, 1.4)
        L.line([(RX0 + 4, y + 14), (RX0 + 4, y + RH - 24)], a, 5)

        L.text(RX0 + 26, y + 16, v["name"], 21, INK, "la", True)
        L.text(RX0 + 26, y + 48, v["en"], 12.5, a, "la", True)
        L.text(RX0 + 130, y + 30, v["one"], 14.5, MUTED, maxw=310)

        for j, (n, note) in enumerate(v["cells"]):
            x0 = C0 + j * COLW
            if n >= 2:
                bg = t
            elif n == 1:
                bg = (252, 252, 253)
            else:
                bg = WHITE
            L.box(x0, y + 6, x0 + COLW - 8, y + RH - 16, 10, bg,
                  li if n >= 2 else (238, 242, 247), 1.2 if n >= 2 else 1.0)
            dots(L, x0 + 22, y + 32, n, a if n else (226, 232, 240))
            L.text(x0 + 22, y + 54, note, 14, SLATE if n else (148, 163, 184),
                   maxw=COLW - 52)

    # ---------------- 第二层：正反馈 ----------------
    FY = RY + 4 * RH + 34
    L.text(56, FY, "第二层 · 它们怎么互相放大", 19, INK, "la", True)
    L.text(56, FY + 34, "单看每一招收益有限；合起来会出现「1+1>3」的正反馈", 15, MUTED)

    chains = [
        ("融合 → 内存 → 吞吐", BLUE, BLUE_T, BLUE_L,
         ["算子融合，中间结果不落地", "显存峰值明显下降", "可以开更大的 batch", "单张图的成本被摊薄"]),
        ("精度 → 带宽 → 融合", ORANGE, ORANGE_T, ORANGE_L,
         ["降到 FP16 / INT8", "要搬的数据量减半", "带宽压力跟着减半", "融合省下的带宽更值钱"]),
        ("内存 → 缓存 → 调优", VIOLET, VIOLET_T, VIOLET_L,
         ["中间张量复用同一块显存", "数据在物理上更紧凑", "L2 命中率大幅提升", "调优选出的 kernel 才跑得动"]),
    ]

    CY0 = FY + 64
    CWc = (2288 - 80) / 3
    for i, (title, c, t, li, steps) in enumerate(chains):
        x0 = 56 + i * (CWc + 40)
        x1 = x0 + CWc
        L.box(x0, CY0, x1, CY0 + 372, 16, WHITE, LINE, 1.6)
        L.text(x0 + 22, CY0 + 18, title, 17, c, "la", True, maxw=CWc - 44)

        for j, st in enumerate(steps):
            yy = CY0 + 58 + j * 78
            L.box(x0 + 20, yy, x1 - 20, yy + 54, 10, t, li, 1.2)
            L.text(x0 + 36, yy + 27, st, 15, SLATE, "lm", maxw=CWc - 76)
            if j < 3:
                L.arrow((x0 + x1) / 2, yy + 56, (x0 + x1) / 2, yy + 74, c, 2.0, 9)

    # ---------------- 底：一句话 ----------------
    by = CY0 + 398
    L.box(56, by, 2344, by + 118, 16, INK, INK, 0)
    L.text(88, by + 20, "面试版一句话", 14, (148, 163, 184), "la", True)
    L.text(88, by + 46,
           "融合减 IO，调优减计算，内存复用减峰值，降精度实现「计算 / 显存 / 搬运」三赢。",
           24, WHITE, "la", True, maxw=2200)
    L.text(88, by + 82, "四招单独用都只是小赚，叠在一起才会出现量级差别。", 17, TEAL_L, "la")

    L.text(56, H - 30, "图 4 · 四招组合拳：各自打在哪、怎么互相放大", 17, FAINT)
    L.text(2344, H - 30, "nxcstudy_ai · 模型格式 · 2026-11", 17, FAINT, "ra")
    L.save(path)


# ======================================================================
# 图 3  推理后端全景
# ======================================================================
BACKENDS = [
    dict(name="NVIDIA 专属", tag="紧跟自家 GPU", accent=TEAL, tint=TEAL_T, light=TEAL_L,
         items=["PyTorch ATen + cuDNN", "Inductor / Triton", "TensorRT", "cuDNN Graph（少量场景）"],
         eat=".pt  ·  .onnx",
         run="NVIDIA GPU",
         note="不换格式直接跑；灵活，但调度最慢"),
    dict(name="跨平台通用", tag="一套代码多硬件", accent=BLUE, tint=BLUE_T, light=BLUE_L,
         items=["ONNX Runtime", "OpenVINO", "Apache TVM", "OpenXLA"],
         eat=".onnx  ·  各自 IR",
         run="CPU / GPU / NPU",
         note="换 EP 就换硬件，模型不用改"),
    dict(name="端侧 / 嵌入式", tag="芯片厂商自带", accent=ORANGE, tint=ORANGE_T, light=ORANGE_L,
         items=["ExecuTorch", "NCNN / MNN", "TFLite / Core ML", "RKNN / QNN / CANN"],
         eat=".pt2 · .onnx · .tflite",
         run="手机 / 车机 / 板卡",
         note="上限由芯片定，算子常缺"),
    dict(name="LLM 运行时", tag="为吞吐设计", accent=VIOLET, tint=VIOLET_T, light=VIOLET_L,
         items=["vLLM", "SGLang", "TensorRT-LLM", "llama.cpp（ggml）"],
         eat="HF 权重  ·  engine",
         run="GPU / CPU / Apple",
         note="重点在调度吞吐，不是单条延迟"),
]


def fig_backends(path):
    W, H = 2400, 1110
    L = Lay(W, H)

    L.text(56, 36, "推理后端：五条路底下真正干活的那一层", 42, INK, "la", True)
    L.text(56, 100,
           "路线是「格式 × 后端」的组合；同一个格式能换后端，同一个后端也能吃多种格式",
           21, MUTED)
    pill(L, "图 3 / 6 · 推理后端")

    # ---------------- 前端 ----------------
    FY0, FY1 = 152, 248
    L.box(56, FY0, 300, FY1, 14, (241, 245, 249), LINE, 1.2)
    L.text(78, FY0 + 24, "前端", 21, INK, "la", True)
    L.text(78, FY0 + 56, "图从哪来", 13.5, MUTED)

    fchips = ["PyTorch eager", "TorchScript", "torch.compile", "ONNX 导出", "TF / JAX"]
    cx = 320
    for i, c in enumerate(fchips):
        w = L.chip(cx, FY0 + 26, c, 17, INK, WHITE, (203, 213, 225), 20, 46, True, 23)
        if i < len(fchips) - 1:
            L.text(cx + w + 12, FY0 + 49, "·", 18, FAINT, "mm")
        cx += w + 34

    L.text(2344, FY0 + 49, "↓ 交给下面任意一个后端", 14, FAINT, "rm")

    # ---------------- 后端四列 ----------------
    CWc = (2288 - 3 * 32) / 4
    CY0, CY1 = 288, 792

    for i, b in enumerate(BACKENDS):
        x0 = 56 + i * (CWc + 32)
        x1 = x0 + CWc
        a, t, li = b["accent"], b["tint"], b["light"]

        L.box(x0, CY0, x1, CY1, 16, WHITE, LINE, 1.6)
        L.box(x0 + 16, CY0 + 16, x1 - 16, CY0 + 88, 12, t, t, 0)
        L.text(x0 + 34, CY0 + 30, b["name"], 20, a, "la", True, maxw=CWc - 60)
        L.text(x0 + 34, CY0 + 60, b["tag"], 13, MUTED, maxw=CWc - 60)

        for j, it in enumerate(b["items"]):
            yy = CY0 + 106 + j * 56
            L.box(x0 + 16, yy, x1 - 16, yy + 46, 10, (250, 251, 253), li, 1.2)
            L.text(x0 + 32, yy + 23, it, 15, SLATE, "lm", maxw=CWc - 60)

        ly = CY0 + 106 + len(b["items"]) * 56 + 6
        L.line([(x0 + 18, ly), (x1 - 18, ly)], HAIR, 1.4)

        L.text(x0 + 22, ly + 10, "吃进", 12.5, FAINT, "la", True)
        L.text(x0 + 72, ly + 10, b["eat"], 14, SLATE, maxw=CWc - 96)
        L.text(x0 + 22, ly + 40, "跑在", 12.5, FAINT, "la", True)
        L.text(x0 + 72, ly + 40, b["run"], 14, SLATE, maxw=CWc - 96)

    # EP 说明挂到第二列底部
    ep_x0 = 56 + 1 * (CWc + 32)
    L.box(ep_x0 + 16, CY1 - 76, ep_x0 + CWc - 16, CY1 - 18, 10, BLUE_T, BLUE_L, 1.2)
    L.text(ep_x0 + 30, CY1 - 62, "EP = 插件", 12.5, BLUE, "la", True)
    L.text(ep_x0 + 30, CY1 - 42,
           "CUDA / TensorRT / OpenVINO / DirectML / QNN", 12.5, SLATE,
           maxw=CWc - 56)

    # 其余三列底部一句话
    for i, b in enumerate(BACKENDS):
        if i == 1:
            continue
        x0 = 56 + i * (CWc + 32)
        L.line([(x0 + 18, CY1 - 70), (x0 + CWc - 18, CY1 - 70)], HAIR, 1.4)
        L.text(x0 + 22, CY1 - 58, "一句话", 12.5, FAINT, "la", True)
        L.text(x0 + 22, CY1 - 34, b["note"], 14, SLATE, maxw=CWc - 44)

    # ---------------- 服务层 ----------------
    SY0, SY1 = 822, 912
    L.box(56, SY0, 300, SY1, 14, (241, 245, 249), LINE, 1.2)
    L.text(78, SY0 + 22, "服务层", 21, INK, "la", True)
    L.text(78, SY0 + 54, "不是后端", 13.5, ROSE)

    schips = [("Triton Inference Server", "多后端挂载 + 动态批处理"),
              ("TorchServe", "PyTorch 官方"),
              ("KServe / Triton on K8s", "云原生部署")]
    cx = 320
    for c, d in schips:
        w = L.chip(cx, SY0 + 22, c, 16, SLATE, WHITE, (203, 213, 225), 18, 42, True, 21)
        L.text(cx + w + 12, SY0 + 43, d, 13, FAINT, "lm")
        cx += w + 24 + L.tw(d, 13) + 40

    # ---------------- 三条要点 ----------------
    by = 942
    L.box(56, by, 2344, by + 118, 14, INK, INK, 0)
    pts = [("① 格式 ≠ 后端", "一个 .onnx 能被 ORT / TensorRT / OpenVINO / TVM 读"),
           ("② EP 是插件", "ORT 的 CUDA EP / TensorRT EP 本质是把别的后端接进来"),
           ("③ 服务层不是后端", "它只管排队和路由，真正算的还是下面那层")]
    for i, (t1, t2) in enumerate(pts):
        x = 88 + i * 756
        L.text(x, by + 22, t1, 17, WHITE, "la", True)
        L.text(x, by + 54, t2, 14, (148, 163, 184), maxw=700)
        L.text(x, by + 86, "", 12, (100, 116, 139))

    L.text(56, H - 30, "图 3 · 推理后端全景：前端 / 后端 / 服务层，三件事不要混", 17, FAINT)
    L.text(2344, H - 30, "nxcstudy_ai · 模型格式 · 2026-11", 17, FAINT, "ra")
    L.save(path)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    fig1(os.path.join(out, "2026-11-推理加速技术路线全景-图2-路线全景.png"))
    fig2(os.path.join(out, "2026-11-推理加速技术路线全景-图1-三堵墙.png"))
    fig3(os.path.join(out, "2026-11-推理加速技术路线全景-图4-ONNX与TensorRT.png"))
    fig_backends(os.path.join(out, "2026-11-推理加速技术路线全景-图3-推理后端全景.png"))
    fig_combo(os.path.join(out, "2026-11-推理加速技术路线全景-图5-四招组合拳.png"))
    fig4(os.path.join(out, "2026-11-推理加速技术路线全景-图6-精度阶梯.png"))
