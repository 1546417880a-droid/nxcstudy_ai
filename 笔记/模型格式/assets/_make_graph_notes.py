# -*- coding: utf-8 -*-
"""生成「算子图」四张图。

图 1  三层视角：模块图 / 算子图 / kernel 图
图 2  解剖：一个节点记什么，一条边记什么
图 3  三格式对照：PyTorch 动态图 / ONNX 静态图 / TensorRT engine
图 4  地址与下发：张量在显存哪、图怎么送到 GPU

用法：python3 _make_graph_notes.py <输出目录>
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
GREY_T = (248, 250, 252)


def pill(L, s):
    w = L.tw(s, 17, True) + 34
    L.chip(2344 - w, 44, s, 17, SLATE, (241, 245, 249), LINE, 17, 40, True)


def band_label(L, y0, y1, name, sub, a, t, li):
    L.box(56, y0, 300, y1, 16, t, li, 1.4)
    cy = (y0 + y1) / 2
    L.text(78, cy - 14, name, 23, a, "lm", True, maxw=210)
    L.text(78, cy + 16, sub, 13.5, MUTED, "lm", maxw=212)


# ======================================================================
# 图 1  三层视角
# ======================================================================
def fig1(path):
    W, H = 2400, 1130
    L = Lay(W, H)

    L.text(56, 36, "同一段模型，三个层次：模块图 / 算子图 / kernel 图", 42, INK, "la", True)
    L.text(56, 100,
           "越往下越接近硬件。平时说的「算子图」，是中间那一层 —— 也是唯一既看不见又不直觉的一层",
           21, MUTED)
    pill(L, "图 1 / 4 · 三层")

    # ---------- 第一层：模块图 ----------
    b1y0, b1y1 = 176, 356
    band_label(L, b1y0, b1y1, "模块图", "代码里的对象树", INK, (241, 245, 249), LINE)

    L.box(320, b1y0, 1240, b1y1, 14, WHITE, (203, 213, 225), 1.4)
    L.text(340, b1y0 + 12, "ResNet50", 15, SLATE, "la", True)
    L.box(344, b1y0 + 40, 1216, b1y1 - 16, 12, GREY_T, (226, 232, 240), 1.2)
    L.text(360, b1y0 + 46, "layer1", 13.5, MUTED, "la", True)
    L.box(368, b1y0 + 70, 1192, b1y1 - 26, 10, WHITE, (226, 232, 240), 1.2)
    L.text(382, b1y0 + 74, "BasicBlock", 12.5, MUTED, "la", True)

    chips = ["conv1", "bn1", "relu", "conv2", "bn2"]
    cx = 384
    for c in chips:
        w = 132
        L.box(cx, b1y0 + 98, cx + w, b1y0 + 138, 9, BLUE_T, BLUE_L, 1.2)
        L.text(cx + w / 2, b1y0 + 118, c, 14, BLUE, "mm", True)
        cx += w + 16

    L.text(1270, b1y0 + 26, "就是你在 Python 里写的结构", 15.5, SLATE)
    L.text(1270, b1y0 + 56, "model.layer1[0].conv1 这样能一层层点下去", 14, MUTED, maxw=1000)
    L.text(1270, b1y0 + 92, "特点：人看得懂，但机器没法直接执行 —— 它只是对象，", 14, MUTED, maxw=1000)
    L.text(1270, b1y0 + 118, "真正跑的时候还要展开成一个个算子", 14, MUTED, maxw=1000)

    # ---------- 第二层：算子图 ----------
    b2y0, b2y1 = 386, 706
    band_label(L, b2y0, b2y1, "算子图", "DAG：节点=算子，边=tensor", BLUE, BLUE_T, BLUE_L)

    ops = [("conv2d", "3×3 / 64"), ("batch_norm", "eps 1e-5"), ("relu", "inplace"),
           ("conv2d", "3×3 / 64"), ("batch_norm", "eps 1e-5"), ("add", "残差"), ("relu", "—")]
    NW, NG = 200, 62
    y_op0, y_op1 = 430, 512
    xs = []
    for i, (nm, attr) in enumerate(ops):
        x = 320 + i * (NW + NG)
        xs.append(x)
        c = BLUE if i != 5 else TEAL
        t = BLUE_T if i != 5 else TEAL_T
        li = BLUE_L if i != 5 else TEAL_L
        L.box(x, y_op0, x + NW, y_op1, 12, t, li, 1.5)
        L.text(x + NW / 2, y_op0 + 20, nm, 17, c, "mm", True)
        L.text(x + NW / 2, y_op0 + 48, attr, 12, MUTED, "mm")
        L.text(x + NW - 10, y_op0 + 8, f"{i + 1}", 11, FAINT, "ra", True)
        if i < len(ops) - 1:
            L.arrow(x + NW + 6, (y_op0 + y_op1) / 2, x + NW + NG - 6, (y_op0 + y_op1) / 2,
                    (203, 213, 225), 2.0, 9)

    # 边上的 shape 标注
    labels = ["1×64×56×56", "1×64×56×56", "1×64×56×56", "1×64×56×56", "1×64×56×56"]
    for i, lb in enumerate(labels):
        L.text(xs[i] + NW + NG / 2, y_op0 - 16, lb, 10.5, FAINT, "mm")

    # 残差跳连
    sx = xs[0] + NW / 2
    L.line([(sx, y_op1), (sx, 592), (xs[5] + NW / 2, 592)], TEAL, 2.0)
    L.arrow(xs[5] + NW / 2, 588, xs[5] + NW / 2, y_op1 + 4, TEAL, 2.0, 9)
    L.text(sx + 16, 570, "identity（跳连）", 12.5, TEAL, "la", True)

    L.text(320, 636, "这一层才是被优化、被调度、被统计的对象：", 14.5, SLATE)
    L.text(320, 664, "融合删的是这里的节点，调优换的是这里的实现，量化改的是这里的精度", 13, MUTED, maxw=1990)

    # ---------- 第三层：kernel 图 ----------
    b3y0, b3y1 = 756, 940
    band_label(L, b3y0, b3y1, "kernel 图", "GPU 上真正执行的序列", TEAL, TEAL_T, TEAL_L)

    ks = [("fused conv+bn+relu", "1 个 kernel 干 3 件事"), ("fused conv+bn", "1 个 kernel 干 2 件事"),
          ("add", "单独一个"), ("relu", "单独一个")]
    kx = 320
    for i, (nm, d) in enumerate(ks):
        w = 400
        L.box(kx, b3y0 + 24, kx + w, b3y0 + 108, 12, TEAL_T, TEAL_L, 1.4)
        L.text(kx + 18, b3y0 + 40, nm, 15, TEAL, "la", True, maxw=w - 36)
        L.text(kx + 18, b3y0 + 72, d, 12.5, SLATE, maxw=w - 36)
        if i < 3:
            L.arrow(kx + w + 6, b3y0 + 66, kx + w + 62, b3y0 + 66, (203, 213, 225), 2.0, 9)
        kx += w + 68

    # ---------- 结论 ----------
    by = 972
    L.box(56, by, 2344, by + 104, 14, INK, INK, 0)
    L.text(88, by + 18, "三层不是一一对应的", 17, WHITE, "la", True)
    L.text(88, by + 50, "一个模块节点会展开成几十个算子；反过来，几个算子又会被融成一个 kernel。"
                        "所以「数一数图上有几个框」这种问题，问哪一层都成立，答案完全不同。",
           15, (148, 163, 184), maxw=2200)
    L.text(88, by + 76, "7 个算子  →  4 个 kernel（因为前两组被融合了）", 15, TEAL_L, "la", True)

    L.text(56, H - 30, "图 1 · 三层视角：模块图 / 算子图 / kernel 图", 17, FAINT)
    L.text(2344, H - 30, "nxcstudy_ai · 模型格式 · 2026-11", 17, FAINT, "ra")
    L.save(path)


# ======================================================================
# 图 2  解剖
# ======================================================================
def fig2(path):
    W, H = 2400, 1250
    L = Lay(W, H)

    L.text(56, 36, "一张算子图里到底记了什么", 42, INK, "la", True)
    L.text(56, 100,
           "节点记「算什么」，边记「传什么」，属性记「怎么算」，地址记「数据在哪」",
           21, MUTED)
    pill(L, "图 2 / 4 · 解剖")

    # ---------- 节点卡 ----------
    NX0, NX1 = 56, 1160
    NY0, NY1 = 172, 720
    L.box(NX0, NY0, NX1, NY1, 18, WHITE, BLUE_L, 1.8)
    L.box(NX0 + 20, NY0 + 18, NX1 - 20, NY0 + 76, 12, BLUE_T, BLUE_T, 0)
    L.text(NX0 + 40, NY0 + 32, "节点 Node", 23, BLUE, "la", True)
    L.text(NX0 + 180, NY0 + 40, "一个算子 = 一次 kernel 调用", 14, MUTED)

    rows = [
        ("op 类型", "aten::conv2d  →  Conv  →  IConvolutionLayer", "决定调哪个 kernel"),
        ("属性", "stride=1  padding=1  dilation=1  groups=1  kernel=3×3", "不进显存，编译期固化"),
        ("输入 ①", "x : [1,64,56,56]  fp16  @0x7f2a_1c00", "激活值，运行时才有地址"),
        ("输入 ②", "w : [64,64,3,3]   fp16  @0x7f8c_4000", "权重，常驻，地址固定"),
        ("输出", "y : [1,64,56,56]  fp16  @0x7f2a_9c00", "下一层的输入"),
        ("拓扑序", "#12", "决定什么时候能开始算"),
    ]
    ry = NY0 + 96
    for i, (k, v, note) in enumerate(rows):
        L.text(NX0 + 34, ry, k, 15, BLUE, "la", True)
        L.text(NX0 + 140, ry, v, 15, SLATE, maxw=760)
        L.text(NX0 + 916, ry, note, 12.5, FAINT, maxw=190)
        if i < len(rows) - 1:
            L.line([(NX0 + 26, ry + 28), (NX1 - 26, ry + 28)], HAIR, 1.2)
        ry += 70

    # ---------- 边卡 ----------
    EX0, EX1 = 1200, 2344
    L.box(EX0, NY0, EX1, NY1, 18, WHITE, TEAL_L, 1.8)
    L.box(EX0 + 20, NY0 + 18, EX1 - 20, NY0 + 76, 12, TEAL_T, TEAL_T, 0)
    L.text(EX0 + 40, NY0 + 32, "边 / Tensor", 23, TEAL, "la", True)
    L.text(EX0 + 220, NY0 + 40, "一条边 = 一块显存 + 一段生命周期", 14, MUTED)

    erows = [
        ("名字", "y", "调试和报错时靠它定位"),
        ("shape", "[1, 64, 56, 56]", "静态图里写死，动态图里运行时才知道"),
        ("dtype", "fp16", "决定一次搬运多少字节"),
        ("layout", "NCHW / NHWC", "同一份数据，不同遍历顺序"),
        ("地址", "0x7f2a_9c00", "kernel 真正读写的位置"),
        ("生命周期", "#12 产生 → #15 消费完 → 可复用", "内存复用的依据"),
    ]
    ry = NY0 + 96
    for i, (k, v, note) in enumerate(erows):
        L.text(EX0 + 34, ry, k, 15, TEAL, "la", True)
        L.text(EX0 + 130, ry, v, 15, SLATE, maxw=620)
        L.text(EX0 + 780, ry, note, 12.5, FAINT, maxw=320)
        if i < len(erows) - 1:
            L.line([(EX0 + 26, ry + 28), (EX1 - 26, ry + 28)], HAIR, 1.2)
        ry += 70

    # ---------- 三格式对照 ----------
    TY = 726
    L.text(56, TY, "同一批信息，三种格式各叫什么", 20, INK, "la", True)

    heads = [("这一项", 56, 380), ("PyTorch（FX Graph）", 380, 1030),
             ("ONNX（GraphProto）", 1030, 1690), ("TensorRT（engine）", 1690, 2344)]
    for nm, x0, x1 in heads:
        L.box(x0, TY + 40, x1, TY + 78, 8, (241, 245, 249), LINE, 1.0)
        L.text(x0 + 16, TY + 59, nm, 13.5, SLATE, "lm", True)

    body = [
        ("算子", "node.op = 'call_function'，target=aten.conv2d", "node.op_type = \"Conv\"", "ILayer，layer.type = kCONVOLUTION"),
        ("参数", "node.args / node.kwargs", "node.attribute（AttributeProto）", "layer 的权重由 builder 内部持有"),
        ("输入输出", "node.all_input_nodes / users", "graph.input / graph.output / initializer", "engine 的 I/O tensor（binding）"),
        ("中间张量", "没有独立对象，只是 node 的输出引用", "graph.value_info（含 shape）", "内部隐式，不对外暴露"),
        ("形状", "proxy 上的 meta['tensor_meta']", "ValueInfoProto.type.tensor_type.shape", "profile 的 min / opt / max"),
        ("怎么读", "graph.print_tabular()", "netron / onnx.load", "trtexec --dumpLayerInfo"),
    ]
    ty = TY + 78
    for i, (a, b, c, d) in enumerate(body):
        h = 58
        fill = WHITE if i % 2 == 0 else GREY_T
        L.box(56, ty, 2344, ty + h, 0, fill, (240, 244, 249), 1.0)
        L.text(72, ty + 18, a, 13.5, INK, "la", True)
        L.text(392, ty + 18, b, 12.5, SLATE, maxw=630)
        L.text(1042, ty + 18, c, 12.5, SLATE, maxw=640)
        L.text(1702, ty + 18, d, 12.5, SLATE, maxw=620)
        ty += h

    L.text(56, ty + 16, "读法：左列是人给的抽象概念，右边三列是它在三种工具里的具体名字。"
                        "换个格式，概念不变，名字全变。", 14, MUTED, maxw=2200)

    L.text(56, H - 30, "图 2 · 解剖：节点、边与三格式字段对照", 17, FAINT)
    L.text(2344, H - 30, "nxcstudy_ai · 模型格式 · 2026-11", 17, FAINT, "ra")
    L.save(path)


# ======================================================================
# 图 3  三格式对照
# ======================================================================
COLS = [
    dict(name="PyTorch 动态图", tag="eager", accent=BLUE, tint=BLUE_T, light=BLUE_L,
         when="没有持久的图", when2="每次 forward 现算",
         shape="任意形状", shape2="运行时才知道",
         flow="Python 原生", flow2="if / for 随便写",
         read="torch.fx.symbolic_trace", read2="graph.print_tabular()",
         merit="灵活、好调试", demerit="几乎没法做全局优化"),
    dict(name="ONNX 静态图", tag="IR", accent=TEAL, tint=TEAL_T, light=TEAL_L,
         when="导出那一刻固定", when2="之后不再变",
         shape="写死，或 dynamic_axes", shape2="声明成符号维",
         flow="只剩 If / Loop 算子", flow2="必须转成图节点",
         read="netron / onnx.load", read2="graph.value_info 看形状",
         merit="可交换、可分析", demerit="导出后想改就得重导"),
    dict(name="TensorRT engine", tag="plan", accent=VIOLET, tint=VIOLET_T, light=VIOLET_L,
         when="编译时固定", when2="和 GPU 型号绑定",
         shape="profile（min/opt/max）", shape2="超出范围要重编",
         flow="基本只剩静态", flow2="分支被裁掉了",
         read="trtexec --dumpLayerInfo", read2="IEngineInspector（JSON）",
         merit="最快，优化到底", demerit="二进制，几乎不可读"),
]


def fig3(path):
    W, H = 2400, 980
    L = Lay(W, H)

    L.text(56, 36, "动态图 / 静态图 / 编译产物：三张完全不同的「图」", 42, INK, "la", True)
    L.text(56, 100,
           "图在什么时刻被确定，决定了它能被优化到什么程度，也决定了你能看见多少",
           21, MUTED)
    pill(L, "图 3 / 4 · 三种图")

    CW, GAP = 736, 40
    Y0, Y1 = 168, 770

    for i, c in enumerate(COLS):
        x0 = 56 + i * (CW + GAP)
        x1 = x0 + CW
        a, t, li = c["accent"], c["tint"], c["light"]

        L.box(x0, Y0, x1, Y1, 18, WHITE, LINE, 1.6)

        # 头
        L.box(x0 + 16, Y0 + 16, x1 - 16, Y0 + 92, 12, t, t, 0)
        L.text(x0 + 36, Y0 + 30, c["name"], 24, INK, "la", True, maxw=CW - 120)
        L.text(x0 + 36, Y0 + 62, c["tag"], 13, a, "la", True)

        items = [("图何时确定", c["when"], c["when2"]),
                 ("形状", c["shape"], c["shape2"]),
                 ("控制流", c["flow"], c["flow2"]),
                 ("怎么查看", c["read"], c["read2"])]
        iy = Y0 + 112
        for j, (k, v1, v2) in enumerate(items):
            L.text(x0 + 26, iy, k, 12.5, FAINT, "la", True)
            L.text(x0 + 26, iy + 24, v1, 16, INK, "la", True, maxw=CW - 60)
            L.text(x0 + 26, iy + 50, v2, 13.5, MUTED, maxw=CW - 60)
            if j < 3:
                L.line([(x0 + 22, iy + 74), (x1 - 22, iy + 74)], HAIR, 1.2)
            iy += 92

        # 优缺点
        L.line([(x0 + 22, iy - 6), (x1 - 22, iy - 6)], (226, 232, 240), 1.4)
        L.text(x0 + 26, iy + 12, "↑", 14, GREEN, "la", True)
        L.text(x0 + 50, iy + 12, c["merit"], 14, GREEN, maxw=CW - 90)
        L.text(x0 + 26, iy + 42, "↓", 14, (190, 18, 60), "la", True)
        L.text(x0 + 50, iy + 42, c["demerit"], 14, (190, 18, 60), maxw=CW - 90)

    # 演化条
    ey = 806
    L.box(56, ey, 2344, ey + 132, 16, INK, INK, 0)
    L.text(88, ey + 18, "一条线看下来", 15, (148, 163, 184), "la", True)

    steps = [("PyTorch 动态图", "能跑就行，不看图"), ("导出 ONNX", "图第一次被固定"),
             ("编译成 engine", "图被改写，追求更快")]
    sx = 88
    for i, (s1, s2) in enumerate(steps):
        L.text(sx, ey + 48, s1, 19, WHITE, "la", True)
        L.text(sx, ey + 82, s2, 14, (148, 163, 184), maxw=560)
        if i < 2:
            ax = sx + 560
            L.arrow(ax, ey + 60, ax + 60, ey + 60, (71, 85, 105), 2.4, 11)
        sx += 700
    L.text(2100, ey + 60, "可读性 ↓　　优化空间 ↑", 15, TEAL_L, "rm", True)

    L.text(56, H - 30, "图 3 · 三种图：谁在什么时候把图定下来", 17, FAINT)
    L.text(2344, H - 30, "nxcstudy_ai · 模型格式 · 2026-11", 17, FAINT, "ra")
    L.save(path)


# ======================================================================
# 图 4  地址与下发
# ======================================================================
def fig4(path):
    W, H = 2400, 1300
    L = Lay(W, H)

    L.text(56, 36, "算子图的两件实物：数据在显存哪个地址、图怎么送到 GPU", 42, INK, "la", True)
    L.text(56, 100,
           "图上画的是「算什么」，跑起来还要回答两个问题：这块内存在哪、这串 kernel 怎么发下去",
           21, MUTED)
    pill(L, "图 4 / 4 · 地址与下发")

    # ================= 左：显存地址 =================
    LX0, LX1 = 56, 1176
    L.box(LX0, 168, LX1, 1000, 18, WHITE, LINE, 1.6)
    L.text(LX0 + 24, 186, "① 张量在显存里的位置", 21, INK, "la", True)
    L.line([(LX0 + 22, 224), (LX1 - 22, 224)], HAIR, 1.4)

    L.text(LX0 + 24, 240, "Tensor 只是「视图」：真正的一块内存叫 storage", 14, MUTED, maxw=1060)

    # storage 条
    sb_y = 292
    L.text(LX0 + 24, sb_y - 20, "storage（一整块连续显存）", 12.5, FAINT, "la", True)
    cells = 16
    cw = (1060 - 20) / cells
    for i in range(cells):
        x = LX0 + 24 + i * cw
        L.box(x, sb_y, x + cw - 2, sb_y + 42, 4, (241, 245, 249), (226, 232, 240), 1.0)
    L.text(LX0 + 24, sb_y + 50, "0x7f2a_0000", 11.5, FAINT, "la")
    L.text(LX1 - 24, sb_y + 50, "0x7f2a_8c00", 11.5, FAINT, "ra")

    views = [("y[0]", 2, 6, BLUE, BLUE_T, BLUE_L, "0x7f2a_1000"),
             ("y[1]", 7, 5, TEAL, TEAL_T, TEAL_L, "0x7f2a_3800"),
             ("y[2]", 12, 4, ORANGE, ORANGE_T, ORANGE_L, "0x7f2a_6000")]
    for nm, st, ln, a, t2, li, addr in views:
        x = LX0 + 24 + st * cw
        w = ln * cw - 2
        L.box(x, sb_y + 2, x + w, sb_y + 40, 4, t2, li, 1.6)
        L.text(x + w / 2, sb_y + 21, nm, 12.5, a, "mm", True)
        L.text(x + w / 2, sb_y + 66, addr, 11, a, "ma")

    L.text(LX0 + 24, 380, "三个 tensor 共用一块 storage，只是 offset / stride 不同 —— 这类张量叫 view，不额外占显存。",
           13.5, SLATE, maxw=1060)

    # 地址复用
    ry = 420
    L.line([(LX0 + 22, ry - 14), (LX1 - 22, ry - 14)], HAIR, 1.4)
    L.text(LX0 + 24, ry, "推理时地址会被复用：生命周期不重叠的 tensor 拿同一块", 14, SLATE, "la", True, maxw=1060)

    slots = ["buf0", "buf1", "buf2", "buf3"]
    gx = LX0 + 96
    gw = 148
    for j, sname in enumerate(slots):
        L.text(gx + j * (gw + 10) + gw / 2, ry + 28, sname, 11.5, FAINT, "ma", True)

    rows = [("t=0", ["x", "y1", None, None], BLUE),
            ("t=1", [None, "y1", "y2", None], TEAL),
            ("t=2", [None, None, "y2", "y3"], ORANGE)]
    for k, (lbl, cells_v, c) in enumerate(rows):
        y = ry + 46 + k * 52
        L.text(LX0 + 24, y + 20, lbl, 12.5, FAINT, "la", True)
        for j, v in enumerate(cells_v):
            x = gx + j * (gw + 10)
            if v is None:
                L.box(x, y, x + gw, y + 40, 6, (252, 252, 253), (238, 242, 247), 1.0)
            else:
                L.box(x, y, x + gw, y + 40, 6, WHITE, c, 1.6)
                L.text(x + gw / 2, y + 20, v, 13, c, "mm", True)

    L.text(LX0 + 24, ry + 212,
           "t=1 的 y2 和 t=0 的 y1 拿到同一块地址 —— 这就是 TensorRT 说的「内存复用」，显存峰值靠它降下来。",
           13, MUTED, maxw=1060)

    # 怎么查
    qy = 672
    L.line([(LX0 + 22, qy - 14), (LX1 - 22, qy - 14)], HAIR, 1.4)
    L.text(LX0 + 24, qy, "实际怎么查", 14, SLATE, "la", True)
    qs = [("PyTorch", "t.data_ptr()   ·   t.storage().data_ptr()   ·   t.stride()"),
          ("TensorRT", "context.setTensorAddress(name, t.data_ptr())"),
          ("显存总览", "torch.cuda.memory_summary() 看池子怎么切的")]
    for j, (a1, a2) in enumerate(qs):
        y = qy + 34 + j * 38
        L.text(LX0 + 24, y, a1, 12.5, BLUE, "la", True)
        L.text(LX0 + 136, y, a2, 13, SLATE, maxw=940)

    # 融合后没有地址
    L.box(LX0 + 22, 880, LX1 - 22, 972, 12, ORANGE_T, ORANGE_L, 1.4)
    L.text(LX0 + 40, 894, "反过来说更重要", 12.5, ORANGE, "la", True)
    L.text(LX0 + 40, 916,
           "被融合掉的中间张量，根本没有地址 —— 它只在寄存器和 shared memory 里短暂存在。",
           13, SLATE, maxw=1020)
    L.text(LX0 + 40, 940, "所以「省显存」和「省带宽」是同一件事的两面。", 13, SLATE, maxw=1020)

    # ================= 右：下发 =================
    RX0, RX1 = 1220, 2344
    L.box(RX0, 168, RX1, 1000, 18, WHITE, LINE, 1.6)
    L.text(RX0 + 24, 186, "② 图怎么送到 GPU", 21, INK, "la", True)
    L.line([(RX0 + 22, 224), (RX1 - 22, 224)], HAIR, 1.4)

    ways = [
        ("Eager 逐算子下发", BLUE, BLUE_T, "每个算子单独调一次 cudaLaunchKernel",
         "CPU 侧开销最大；kernel 之间有缝隙"),
        ("图执行器（ORT / TorchScript）", TEAL, TEAL_T, "先遍历整张图，再按序挨个 launch",
         "省掉 Python 与派发开销，仍然是 N 次 launch"),
        ("TensorRT enqueue", VIOLET, VIOLET_T, "一次 enqueue，GPU 内部跑完整个 plan",
         "CPU 只发一次命令；kernel 序列已经排好"),
        ("CUDA Graph replay", ORANGE, ORANGE_T, "录制一次全部 launch，之后一键重放",
         "地址被写死，必须 copy 进同一块 static buffer"),
    ]
    wy = 244
    for i, (nm, c, t, d1, d2) in enumerate(ways):
        L.box(RX0 + 22, wy, RX1 - 22, wy + 176, 12, t, t, 0)
        L.text(RX0 + 42, wy + 14, nm, 17, c, "la", True, maxw=RX1 - RX0 - 84)
        L.text(RX0 + 42, wy + 44, d1, 13.5, SLATE, maxw=RX1 - RX0 - 96)
        L.text(RX0 + 42, wy + 74, d2, 13, MUTED, maxw=RX1 - RX0 - 96)

        # 小示意图：N 个方块
        n = 4 if i >= 2 else 5
        kx = RX0 + 42
        for j in range(n):
            kw = 44
            L.box(kx, wy + 106, kx + kw, wy + 144, 5, WHITE, c, 1.4)
            L.text(kx + kw / 2, wy + 125, "k", 11, c, "mm", True)
            kx += kw + 8
        if i == 2:
            L.text(kx + 6, wy + 125, "…一次调用全部排好", 11.5, c, "lm", True)
        if i == 3:
            L.text(kx + 6, wy + 125, "…replay 无 CPU 介入", 11.5, c, "lm", True)
        wy += 190

    # ================= 底：坑 =================
    by = 1030
    L.box(56, by, 2344, by + 106, 14, INK, INK, 0)
    L.text(88, by + 16, "最容易踩的坑：CUDA Graph 把地址焊死了", 16, WHITE, "la", True)
    L.text(88, by + 48,
           "capture 的那一刻，每个 kernel 读写的地址被记进图里。之后 replay 时如果喂一块新显存，"
           "GPU 还是会去读旧地址，结果静默错误。",
           14.5, (148, 163, 184), maxw=2200)
    L.text(88, by + 78, "所以必须：预留一块 static buffer → 每次把新数据 copy 进去 → replay → 从同一块读出结果。",
           14.5, TEAL_L, maxw=2200)

    L.text(56, H - 30, "图 4 · 地址与下发：从算子图到 GPU 执行", 17, FAINT)
    L.text(2344, H - 30, "nxcstudy_ai · 模型格式 · 2026-11", 17, FAINT, "ra")
    L.save(path)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    S = "2026-11-算子图"
    fig1(os.path.join(out, f"{S}-图1-三层视角.png"))
    fig2(os.path.join(out, f"{S}-图2-节点与边.png"))
    fig3(os.path.join(out, f"{S}-图3-三种图.png"))
    fig4(os.path.join(out, f"{S}-图4-地址与下发.png"))
