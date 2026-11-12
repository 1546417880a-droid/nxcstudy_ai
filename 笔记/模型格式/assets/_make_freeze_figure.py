# -*- coding: utf-8 -*-
"""生成「算子图 · 冻结程度」图（图 5）。

一张表 = 节点 / 边 / 属性 / 地址 / 实现。
四种图 = 这张表在四个时刻的快照：每一格是「活」还是「死」。

用法：python3 _make_freeze_figure.py <输出目录>
"""
import os
import sys

from _lay import (
    Lay, INK, MUTED, FAINT, LINE, HAIR, BG, WHITE, SLATE,
    BLUE, BLUE_T, BLUE_L, GREEN, GREEN_T, GREEN_L,
    ORANGE, ORANGE_T, ORANGE_L,
)

# ---------- 四种格状态 ----------
RUNTIME = dict(fill=(250, 250, 251), line=LINE, accent=MUTED,
               main=SLATE, name="运行时决定")
SYMBOL = dict(fill=BLUE_T, line=BLUE_L, accent=BLUE, main=BLUE,
              name="符号 / 只声明")
FROZEN = dict(fill=GREEN_T, line=GREEN_L, accent=GREEN, main=GREEN,
              name="编译期写死")
REWRITE = dict(fill=ORANGE_T, line=ORANGE_L, accent=ORANGE, main=ORANGE,
               name="被改写 / 消失")
ABSENT = dict(fill=WHITE, line=(226, 232, 240), accent=FAINT, main=FAINT,
              name="表里没有")
STATES = [RUNTIME, SYMBOL, FROZEN, REWRITE, ABSENT]


def wrap(L, s, sz, bold, maxw):
    out, cur = [], ""
    for ch in s:
        if cur and L.tw(cur + ch, sz, bold) > maxw:
            out.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        out.append(cur)
    return out


def pill(L, s):
    w = L.tw(s, 17, True) + 34
    L.chip(2352 - w, 44, s, 17, SLATE, (241, 245, 249), LINE, 17, 40, True)


def cell(L, x0, y0, w, h, st, status, main, sub):
    L.box(x0, y0, x0 + w, y0 + h, 12, st["fill"], st["line"], 1.5)
    L.d.ellipse([(x0 + 18) * 2, (y0 + 20) * 2, (x0 + 28) * 2, (y0 + 30) * 2],
                fill=st["accent"])
    L.text(x0 + 38, y0 + 25, status, 13, st["accent"], "lm", True)
    L.text(x0 + 22, y0 + 58, main, 18, st["main"], "la", True, maxw=w - 44)
    lines = wrap(L, sub, 13.5, False, w - 44)[:2]
    for i, ln in enumerate(lines):
        L.text(x0 + 22, y0 + 94 + i * 21, ln, 13.5, MUTED)


def col(L, x, title, desc, cells, freeze, cost, c, cl):
    w = 500
    L.box(x, 232, x + w, 352, 14, cl, c, 1.6)
    L.text(x + 24, 250, title, 24, c, "la", True)
    for i, ln in enumerate(wrap(L, desc, 13.5, False, w - 48)[:2]):
        L.text(x + 24, 290 + i * 21, ln, 13.5, MUTED)

    y = 372
    for st, status, main, s in cells:
        cell(L, x, y, w, 134, st, status, main, s)
        y += 144

    # 冻结时刻 + 代价
    L.box(x, 1108, x + w, 1224, 12, WHITE, LINE, 1.3)
    L.text(x + 22, 1128, "冻结时刻", 12.5, FAINT, "la", True)
    for i, ln in enumerate(wrap(L, freeze, 15.5, True, w - 44)[:1]):
        L.text(x + 22, 1151, ln, 15.5, INK, "la", True)
    L.text(x + 22, 1180, "代价 · " + cost, 13, MUTED, "la", maxw=w - 44)


def legend(L):
    x = 56
    for st in STATES:
        w = L.tw(st["name"], 15, True) + 62
        L.box(x, 158, x + w, 202, 12, st["fill"], st["line"], 1.4)
        L.d.ellipse([(x + 16) * 2, (174) * 2, (x + 26) * 2, (184) * 2], fill=st["accent"])
        L.text(x + 34, 180, st["name"], 15, st["main"], "lm", True)
        x += w + 16


def main(outdir):
    W, H = 2400, 1600
    L = Lay(W, H)

    L.text(56, 36, "同一张表，四个时刻的快照：哪几列已经写死", 42, INK, "la", True)
    L.text(56, 100,
           "算子图 = 一张表。表里的每一格，要么「运行时才知道」，要么「编译期就定死」——两种图的区别就是这个",
           20, MUTED, maxw=2200)
    pill(L, "图 5 · 冻结程度")
    legend(L)

    # 左侧行标签
    rows = [
        ("行 · 结构", "有谁，谁接谁"),
        ("属性", "stride / pad"),
        ("shape", "多大"),
        ("地址", "数据在哪"),
        ("实现", "哪个 kernel"),
    ]
    for i, (a, b) in enumerate(rows):
        y = 372 + i * 144
        L.text(56, y + 40, a, 19, INK, "la", True)
        L.text(56, y + 70, b, 13.5, MUTED, "la", maxw=170)
        if i < len(rows) - 1:
            L.dash(56, y + 134 + 5, 238, y + 134 + 5, (226, 232, 240), 1.2, 5, 5)

    cols = [
        ("PyTorch 动态图", "表不存在，每次 forward 现写一张，用完就扔", [
            (RUNTIME, "每次现算", "由 Python 控制流决定", "if / for 直接改行数"),
            (RUNTIME, "每次现算", "写在调用参数里", "nn.Conv2d(stride=1)"),
            (RUNTIME, "每次现算", "真实输入才知道", "拿到数据才算得出来"),
            (RUNTIME, "每次现填", "分配器随手给一块", "缓存池复用，地址会变"),
            (RUNTIME, "每次现选", "按 shape / dtype 挑", "同算子不同输入不同 kernel"),
        ], "没有 —— 表用完即弃", "每次 forward 付 Python 派发", INK, (241, 245, 249)),

        ("FX 图", "表被抓出来，成了可以被编译器改写的对象", [
            (FROZEN, "写死 @ trace", "控制流一变就断", "graph break → 掉回 eager"),
            (FROZEN, "写死", "记进 args / kwargs", "成了可遍历的字段"),
            (SYMBOL, "符号 SymInt", "一张图跑多种尺寸", "动态 shape 还活着"),
            (ABSENT, "表里没有", "图只描述算法", "地址要等执行时才谈"),
            (RUNTIME, "compile 时生成", "交给后端出 kernel", "Triton 在此环节介入"),
        ], "trace / compile 那一刻", "控制流一改，整张表作废", BLUE, BLUE_T),

        ("ONNX", "表被存档成通用格式，谁都能读、谁都改不了", [
            (FROZEN, "写死 @ 导出", "算子 1:1 映射", "od.op_type 定死"),
            (FROZEN, "写死", "node.attribute", "跨框架可读的同一套字段"),
            (FROZEN, "写死", "或 dynamic_axes 留符号维", "dummy input 猜出来的"),
            (ABSENT, "表里没有", "initializer 只有权重值", "谁执行谁负责分配"),
            (ABSENT, "表里没有", "只有 op_type，没有实现", "实现留给下游引擎"),
        ], "导出那一刻（快照）", "改一行结构就要重导", GREEN, GREEN_T),

        ("TensorRT engine", "表被改写 + 特化，只剩一份执行计划", [
            (REWRITE, "改写 @ build", "融合 → 行被删掉", "conv+bn+relu 合成一层"),
            (FROZEN, "编译进常量", "烧进 kernel", "stride 成了代码不是参数"),
            (FROZEN, "写死成区间", "profile min / opt / max", "超区间就得重编"),
            (FROZEN, "编译期规划", "offset 复用 + 运行时绑指针", "峰值显存是算出来的"),
            (FROZEN, "写死并特化", "选好 kernel 与精度", "FP16 / INT8 在此时定"),
        ], "build 那一刻（绑 GPU 型号）", "换卡 / 升 TRT / 超 profile → 重编", ORANGE, ORANGE_T),
    ]

    for i, (t, sub, cells, freeze, cost, c, cl) in enumerate(cols):
        col(L, 250 + i * 534, t, sub, cells, freeze, cost, c, cl)

    # ---------- 底部：怎么读 ----------
    L.line([(56, 1258), (2352, 1258)], (226, 232, 240), 1.6)

    L.text(56, 1284, "三个读法", 20, INK, "la", True)
    reads = [
        ("① 竖着看一列", "这一种图，把表冻到了什么程度。第一列全灰，最后一列全绿——就是「写死程度」。"),
        ("② 横着看一行", "同一类信息在流水线上越来越死：结构最先死，地址最晚死，实现死得最彻底。"),
        ("③ 别当进度条看", "FX 的 shape 比 ONNX 更活，ONNX 的地址整列是空的——写死是逐格的，不是一维的。"),
    ]
    for i, (a, b) in enumerate(reads):
        y = 1330 + i * 78
        L.text(56, y, a, 16.5, SLATE, "la", True)
        L.text(210, y, b, 15.5, MUTED, "la", maxw=900)

    L.box(1160, 1284, 2352, 1570, 16, WHITE, LINE, 1.4)
    L.text(1192, 1306, "一张表格换来的东西", 17, SLATE, "la", True)
    trades = [
        ("结构写死", "才能融合 —— 得先知道谁接谁"),
        ("形状写死", "才能规划内存 —— 活跃区间算地址复用"),
        ("地址写死", "才能零拷贝 / CUDA Graph / 一次 enqueue"),
        ("实现写死", "CPU 只发一次命令，GPU 一口气跑完"),
    ]
    for i, (a, b) in enumerate(trades):
        y = 1350 + i * 44
        L.text(1192, y, a, 14.5, GREEN, "la", True)
        L.text(1310, y, "→  " + b, 14.5, MUTED, "la", maxw=1010)
    L.line([(1192, 1508), (2320, 1508)], (226, 232, 240), 1.2)
    L.text(1192, 1526, "每冻一列，换一类优化，丢一类灵活性。", 15.5, INK, "la", True)

    out = os.path.join(outdir, "2026-11-算子图-图5-冻结程度.png")
    L.save(out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
