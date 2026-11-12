# -*- coding: utf-8 -*-
"""生成「空间同步 / 时间同步如何穿插到各节点」两张图。

图 1  双轨穿插：7 个处理节点 × 空间轨 / 时间轨，青线穿过的是耦合点
图 2  三个耦合点放大：去畸变 / 融合关联 / 控制延迟补偿

沿用 _make_localization.py 的 Lay 画布（PIL 直绘，2× 超采样，无浏览器依赖）。

用法：python3 _make_spacetime_sync.py <输出目录>
"""
import os
import sys

from _make_localization import (
    Lay, INK, MUTED, FAINT, LINE, HAIR, BG, WHITE, SLATE,
    BLUE, BLUE_T, BLUE_L, GREEN, GREEN_T, GREEN_L,
    ORANGE, ORANGE_T, ORANGE_L, TEAL, TEAL_T, TEAL_L,
)

S = 2
VIOLET = (109, 40, 217)
VIOLET_T = (245, 243, 255)
VIOLET_L = (221, 214, 254)

# ======================================================================
# 图 1  双轨穿插
# ======================================================================

# (节点名, 频率, 空间关键词, [空间两行], 时间关键词, [时间两行], 是否耦合)
NODES = [
    ("驱动 / 接入", "各传感器原始率",
     "外参表", ["加载标定文件", "建静态 tf 树"],
     "打戳", ["标采集时刻", "统一时钟域"], False),

    ("预处理", "10–20 Hz",
     "内参", ["畸变模型校正", "ROI / 地面分割"],
     "逐点时间", ["按 t_i 补偿运动", "对齐到帧参考刻"], True),

    ("单传感器检测", "10–20 Hz",
     "转系", ["结果转车体系", "lidar → base_link"],
     "本帧", ["只取本帧 stamp", "不跨源对齐"], False),

    ("多传感器融合", "10 Hz",
     "投影", ["K[R|t] 投影", "跨模态空间关联"],
     "软同步", ["插值到同一刻", "最近邻 / 线性"], True),

    ("目标跟踪", "10 Hz",
     "同系", ["同一坐标系下", "才算同一目标"],
     "帧间 dt", ["时间间隔", "航迹外推门限"], True),

    ("轨迹预测", "10 Hz",
     "转地图", ["目标系 → 地图系", "给规划绝对位置"],
     "历史窗", ["回看 3–5 s", "时序建模"], False),

    ("规划 / 控制", "10–50 Hz",
     "tf 链", ["车体 → 世界", "全链贯通"],
     "延迟", ["补偿计算+执行", "对齐控制周期"], True),
]


def fig1(path):
    W, H = 2400, 1010
    L = Lay(W, H)

    # ---------------- 头部 ----------------
    L.text(56, 36, "空间同步与时间同步，怎么插进每一个节点", 42, INK, "la", True)
    L.text(56, 100,
           "两条轨并行贯穿整条链路：一条管「在哪」，一条管「何时」。七个节点里有四个必须两条同时到位",
           21, MUTED)

    t = "图 1 / 2 · 双轨穿插"
    tw = L.tw(t, 17, True) + 34
    L.chip(2344 - tw, 44, t, 17, SLATE, (241, 245, 249), LINE, 17, 40, True)

    # ---------------- 图例 ----------------
    ly = 158
    lx = 56
    for label, col, kind in [
        ("空间轨 · 外参 / 内参 / 坐标链", BLUE, "line"),
        ("时间轨 · 时钟 / 时间戳 / 插值", ORANGE, "line"),
        ("青线穿过节点 = 双轨耦合，缺一失效", TEAL, "thread"),
    ]:
        if kind == "line":
            L.line([(lx, ly), (lx + 34, ly)], col, 3.2)
            lx += 44
        else:
            L.line([(lx + 17, ly - 16), (lx + 17, ly + 16)], col, 3.2)
            lx += 44
        L.text(lx, ly, label, 16.5, MUTED, "lm")
        lx += L.tw(label, 16.5) + 46

    L.text(2306, ly, "数据流", 15.5, FAINT, "rm")
    L.arrow(2318, ly, 2344, ly, FAINT, 1.6, 8)

    # ---------------- 几何 ----------------
    CX0, CX1 = 320, 2340
    N = len(NODES)
    GAP = 18
    CW = (CX1 - CX0 - GAP * (N - 1)) / N

    SY0, SY1 = 214, 334          # 空间卡
    NY0, NY1 = 382, 502          # 节点
    TY0, TY1 = 550, 670          # 时间卡

    # ---------------- 左侧行标 ----------------
    def lane(y0, y1, name, sub, a, t, li):
        L.box(56, y0, 296, y1, 15, t, li, 1.4)
        cy = (y0 + y1) / 2
        L.text(78, cy - 14, name, 23, a, "lm", True, maxw=196)
        L.text(78, cy + 16, sub, 14.5, MUTED, "lm", maxw=200)

    lane(SY0, SY1, "空间轨", "在哪 · 坐标系", BLUE, BLUE_T, BLUE_L)
    lane(NY0, NY1, "处理节点", "谁在用 · 做什么", INK, (241, 245, 249), LINE)
    lane(TY0, TY1, "时间轨", "何时 · 时刻对齐", ORANGE, ORANGE_T, ORANGE_L)

    # ---------------- 先画耦合线（被卡片盖住，只留缝隙） ----------------
    for i, nd in enumerate(NODES):
        if not nd[6]:
            continue
        x = CX0 + i * (CW + GAP)
        cx = x + CW / 2
        L.line([(cx, SY0 + 14), (cx, TY1 - 14)], TEAL, 3.0)

    # ---------------- 卡片 ----------------
    for i, (name, freq, st, sbody, tt, tbody, coupled) in enumerate(NODES):
        x = CX0 + i * (CW + GAP)
        cx = x + CW / 2
        accent = TEAL if coupled else LINE

        # 空间卡
        L.box(x, SY0, x + CW, SY1, 13, BLUE_T, BLUE_L, 1.3)
        L.text(x + 16, SY0 + 12, st, 19, BLUE, "la", True, maxw=CW - 32)
        L.text(x + 16, SY0 + 48, sbody[0], 15, SLATE, maxw=CW - 28)
        L.text(x + 16, SY0 + 76, sbody[1], 15, SLATE, maxw=CW - 28)

        # 时间卡
        L.box(x, TY0, x + CW, TY1, 13, ORANGE_T, ORANGE_L, 1.3)
        L.text(x + 16, TY0 + 12, tt, 19, ORANGE, "la", True, maxw=CW - 32)
        L.text(x + 16, TY0 + 48, tbody[0], 15, SLATE, maxw=CW - 28)
        L.text(x + 16, TY0 + 76, tbody[1], 15, SLATE, maxw=CW - 28)

        # 节点
        L.box(x, NY0, x + CW, NY1, 15, WHITE,
              TEAL if coupled else (203, 213, 225),
              2.6 if coupled else 1.6)
        L.line([(x + 5, NY0 + 20), (x + 5, NY1 - 20)], accent, 5)
        L.text(x + 22, NY0 + 20, name, 19.5, INK, "la", True, maxw=CW - 56)
        L.text(x + 22, NY0 + 58, freq, 14.5, MUTED, maxw=CW - 40)
        L.text(x + CW - 14, NY0 + 20, f"{i + 1:02d}", 14, FAINT, "ra", True)

        # 上下连接箭头
        if coupled:
            L.arrow(cx, SY1 + 2, cx, NY0 - 4, TEAL, 2.6, 11)
            L.arrow(cx, NY1 + 2, cx, TY0 - 4, TEAL, 2.6, 11)
        else:
            L.dash(cx, SY1 + 2, cx, NY0 - 10, (203, 213, 225), 1.6, 7, 6)
            L.arrow(cx, NY0 - 12, cx, NY0 - 4, (203, 213, 225), 1.6, 8)
            L.dash(cx, NY1 + 2, cx, TY0 - 10, (203, 213, 225), 1.6, 7, 6)
            L.arrow(cx, TY0 - 12, cx, TY0 - 4, (203, 213, 225), 1.6, 8)

    # 数据流方向已在图例行右侧标注

    # ---------------- 结论条 ----------------
    by0 = 726
    L.box(56, by0, 2344, by0 + 142, 18, INK, INK, 0)
    L.text(88, by0 + 22, "七个节点，四个是耦合点", 26, WHITE, "la", True)
    L.text(88, by0 + 64, "② 预处理　④ 融合　⑤ 跟踪　⑦ 规划控制", 17.5, (148, 163, 184))
    L.text(88, by0 + 96, "只有一条轨到位，产出就是废数据 —— 而不是「精度差一点」", 16, TEAL_L)

    stats = [
        ("2.8 m", "时间差 100 ms @ 100 km/h"),
        ("0.87 m", "外参差 1° @ 50 m"),
        ("5.5 m", "同步容差 0.2 s @ 100 km/h"),
    ]
    sx = 1180
    for val, lab in stats:
        L.line([(sx - 34, by0 + 26), (sx - 34, by0 + 116)], (45, 58, 78), 1.4)
        L.text(sx, by0 + 30, val, 27, WHITE, "la", True)
        L.text(sx, by0 + 74, lab, 15, (148, 163, 184), maxw=300)
        sx += 392

    # ---------------- 语义脚注 ----------------
    ny0 = 902
    L.box(56, ny0, 2344, ny0 + 62, 13, (250, 250, 252), LINE, 1.2)
    L.chip(74, ny0 + 13, "前置条件", 15, VIOLET, VIOLET_T, VIOLET_L, 14, 36, True)
    L.text(196, ny0 + 31,
           "两条轨能不能真的对上，取决于第三件事 —— 语义：时间戳标的是曝光中点还是读出完成？原点在后轴中心还是质心？单位是 m 还是 mm？",
           16.5, SLATE, "lm", maxw=2100)

    L.text(56, H - 34, "图 1 · 双轨穿插：空间轨在上、时间轨在下，青线穿过的节点必须两条同时到位", 17, FAINT)
    L.text(2344, H - 34, "nxcstudy_ai · 三维视觉 · 2026-11", 17, FAINT, "ra")
    L.save(path)


# ======================================================================
# 图 2  三个耦合点放大
# ======================================================================
PANELS = [
    dict(idx="A", name="去畸变", where="预处理 · 10–20 Hz", accent=BLUE, tint=BLUE_T,
         spa=("空间输入", "外参 T_base←lidar"),
         tim=("时间输入", "逐点时间戳 t_i"),
         op="运动补偿 undistort",
         out="统一到 t_ref 的车体系点云",
         fail=["缺外参：点云整体贴在错的位姿上",
               "缺点时刻：高速时点云被沿运动方向拉长"],
         chk="匀速直线 + 路边杆状物，看点云是否被拉长"),

    dict(idx="B", name="融合关联", where="多传感器融合 · 10 Hz", accent=TEAL, tint=TEAL_T,
         spa=("空间输入", "投影 K[R|t]"),
         tim=("时间输入", "软同步 + 插值"),
         op="关联 association",
         out="同一物体、同一时刻的候选对",
         fail=["缺空间：框被投到隔壁车道",
               "缺时间：把相隔 0.2 s 的目标当成同一个"],
         chk="点云叠到图像上对远处杆物，肉眼可辨"),

    dict(idx="C", name="控制延迟补偿", where="规划 → 控制 · 50 Hz", accent=ORANGE, tint=ORANGE_T,
         spa=("空间输入", "车体 → 世界 tf"),
         tim=("时间输入", "计算 + 执行延迟 τ"),
         op="位姿外推 + 时间对齐",
         out="在正确时刻生效的指令",
         fail=["缺空间：方向打在错的路上",
               "缺时间：高速时车身画龙"],
         chk="给一个阶跃转向，看是否过冲、是否画龙"),
]


def fig2(path):
    W, H = 2400, 930
    L = Lay(W, H)

    L.text(56, 36, "三个耦合点：同一段代码里，两条轨必须同时到位", 42, INK, "la", True)
    L.text(56, 100,
           "它们都不是「先做完空间对齐，再做时间对齐」，而是在同一个函数里同时用上两套参数",
           21, MUTED)

    t = "图 2 / 2 · 耦合点"
    tw = L.tw(t, 17, True) + 34
    L.chip(2344 - tw, 44, t, 17, SLATE, (241, 245, 249), LINE, 17, 40, True)

    PW, PG = 736, 40
    Y0, Y1 = 172, 760
    for pi, p in enumerate(PANELS):
        x0 = 56 + pi * (PW + PG)
        x1 = x0 + PW
        a, t2 = p["accent"], p["tint"]
        cx = (x0 + x1) / 2

        L.box(x0, Y0, x1, Y1, 18, WHITE, LINE, 1.6)

        # 标题
        L.text(x0 + 24, Y0 + 18, p["idx"], 19, a, "la", True)
        L.text(x0 + 56, Y0 + 18, p["name"], 25, INK, "la", True, maxw=440)
        L.text(x0 + 24, Y0 + 58, p["where"], 15.5, MUTED)
        L.line([(x0 + 22, Y0 + 88), (x1 - 22, Y0 + 88)], HAIR, 1.4)

        # 两路输入（并排）
        iy = Y0 + 104
        hw = (PW - 48 - 16) / 2
        for j, (label, val, col, bgc, lc) in enumerate([
            (p["spa"][0], p["spa"][1], BLUE, BLUE_T, BLUE_L),
            (p["tim"][0], p["tim"][1], ORANGE, ORANGE_T, ORANGE_L),
        ]):
            bx0 = x0 + 24 + j * (hw + 16)
            bx1 = bx0 + hw
            L.box(bx0, iy, bx1, iy + 72, 11, bgc, lc, 1.3)
            L.text(bx0 + 16, iy + 13, label, 13, col, "la", True)
            L.text(bx0 + 16, iy + 40, val, 16.5, SLATE, maxw=hw - 28)

        # 汇聚总线
        lx_c = x0 + 24 + hw / 2
        rx_c = x0 + 24 + hw + 16 + hw / 2
        bus_y = iy + 96
        L.line([(lx_c, iy + 72), (lx_c, bus_y)], BLUE, 2.0)
        L.line([(rx_c, iy + 72), (rx_c, bus_y)], ORANGE, 2.0)
        L.line([(lx_c, bus_y), (rx_c, bus_y)], (203, 213, 225), 2.0)
        L.arrow(cx, bus_y, cx, bus_y + 18, INK, 2.2, 10)

        # 变换
        oy = bus_y + 22
        L.box(x0 + 24, oy, x1 - 24, oy + 62, 12, INK, INK, 0)
        L.text(cx, oy + 31, p["op"], 21, WHITE, "mm", True, maxw=PW - 80)

        # 输出
        L.arrow(cx, oy + 66, cx, oy + 90, a, 2.2, 10)
        uy = oy + 94
        L.box(x0 + 24, uy, x1 - 24, uy + 58, 12, t2, a, 1.6)
        L.text(cx, uy + 29, p["out"], 17, INK, "mm", True, maxw=PW - 72)

        # 失效
        fy = uy + 86
        L.line([(x0 + 22, fy - 14), (x1 - 22, fy - 14)], HAIR, 1.4)
        L.text(x0 + 24, fy, "缺一条会怎样", 15, (190, 18, 60), "la", True)
        for j, s in enumerate(p["fail"]):
            L.text(x0 + 24, fy + 30 + j * 29, "·", 16, (190, 18, 60))
            L.text(x0 + 40, fy + 30 + j * 29, s, 15, SLATE, maxw=PW - 80)

        # 怎么验证
        vy = fy + 112
        L.line([(x0 + 22, vy - 14), (x1 - 22, vy - 14)], HAIR, 1.4)
        L.text(x0 + 24, vy, "怎么验证", 15, GREEN, "la", True)
        L.text(x0 + 24, vy + 30, p["chk"], 15, SLATE, maxw=PW - 56)

    # 底部结论
    by0 = H - 118
    L.box(56, by0, 2344, by0 + 50, 12, (250, 250, 252), LINE, 1.2)
    L.text(78, by0 + 25,
           "共同点：空间和时间在这三个点上不是两个步骤，而是一次联合求解 —— 所以「先对齐时间再对齐空间」在代码里根本落不了地。",
           16.5, SLATE, "lm", maxw=2240)

    L.text(56, H - 30, "图 2 · 三个耦合点：去畸变 / 融合关联 / 控制延迟补偿", 17, FAINT)
    L.text(2344, H - 30, "nxcstudy_ai · 三维视觉 · 2026-11", 17, FAINT, "ra")
    L.save(path)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    fig1(os.path.join(out, "2026-11-时空同步-图1-双轨穿插.png"))
    fig2(os.path.join(out, "2026-11-时空同步-图2-三个耦合点.png"))
