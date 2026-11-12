# -*- coding: utf-8 -*-
"""生成「自动驾驶定位」两张图（PIL 直绘，2× 超采样，无浏览器依赖）。

图 1  技术全景：四类定位原理并排 + 场景选型
图 2  分层架构：传感器 → 前端解算 → 融合 → 输出 的数据流

用法：python3 _make_localization.py <输出目录>
"""
import os
import sys
import math
from PIL import Image, ImageDraw, ImageFont

S = 2
EXPORT_SCALE = 1.3   # 2400 设计宽 → 3120 像素，与笔记里其它图一致
FONT = "/System/Library/Fonts/Hiragino Sans GB.ttc"

INK = (15, 23, 42)
MUTED = (100, 116, 139)
FAINT = (148, 163, 184)
LINE = (226, 232, 240)
HAIR = (241, 245, 249)
BG = (248, 250, 252)
WHITE = (255, 255, 255)
SLATE = (51, 65, 85)

BLUE = (37, 99, 235)
BLUE_T = (238, 245, 255)
BLUE_L = (191, 219, 254)
GREEN = (5, 130, 95)
GREEN_T = (236, 253, 245)
GREEN_L = (167, 243, 208)
ORANGE = (190, 62, 10)
ORANGE_T = (255, 246, 236)
ORANGE_L = (253, 210, 175)
TEAL = (13, 137, 128)
TEAL_T = (238, 251, 250)
TEAL_L = (150, 232, 224)

_fc = {}


def font(sz, bold=False):
    k = (round(sz * 4), bold)
    if k not in _fc:
        _fc[k] = ImageFont.truetype(FONT, max(6, int(round(sz * S))), index=2 if bold else 0)
    return _fc[k]


class Lay:
    def __init__(self, w, h):
        self.im = Image.new("RGBA", (int(w * S), int(h * S)), BG + (255,))
        self.d = ImageDraw.Draw(self.im)
        self.w, self.h = w, h
        self.fits = []

    # ---------- 基础图元 ----------
    def tw(self, s, sz, bold=False):
        return self.d.textlength(s, font=font(sz, bold)) / S

    def text(self, x, y, s, sz=22, color=INK, anchor="la", bold=False, maxw=None):
        if maxw is not None and self.tw(s, sz, bold) > maxw:
            while sz > 9 and self.tw(s, sz, bold) > maxw:
                sz -= 0.5
            self.fits.append((s[:14], round(sz, 1)))
        self.d.text((x * S, y * S), s, font=font(sz, bold), fill=color, anchor=anchor)

    def vtext(self, x, y, s, sz, color, bold=False):
        """逆时针旋转 90° 的竖排文字，中心在 (x, y)，自下而上阅读。"""
        f = font(sz, bold)
        w = self.d.textlength(s, font=f) / S + 12
        h = sz * 2.2
        tmp = Image.new("RGBA", (int(w * S), int(h * S)), (0, 0, 0, 0))
        ImageDraw.Draw(tmp).text((6 * S, h * S / 2), s, font=f, fill=color, anchor="lm")
        rot = tmp.rotate(90, expand=True)
        self.im.alpha_composite(rot, (int(x * S - rot.width / 2), int(y * S - rot.height / 2)))

    def box(self, x0, y0, x1, y1, r=16, fill=WHITE, outline=LINE, width=1.6):
        self.d.rounded_rectangle([x0 * S, y0 * S, x1 * S, y1 * S], radius=r * S,
                                 fill=fill, outline=outline, width=max(1, int(width * S)))

    def line(self, pts, color, width=2):
        self.d.line([(x * S, y * S) for x, y in pts], fill=color, width=max(1, int(width * S)), joint="curve")

    def dash(self, x0, y0, x1, y1, color, width=2, on=9, off=7):
        L = math.hypot(x1 - x0, y1 - y0)
        if L == 0:
            return
        ux, uy = (x1 - x0) / L, (y1 - y0) / L
        t = 0.0
        while t < L:
            e = min(t + on, L)
            self.line([(x0 + ux * t, y0 + uy * t), (x0 + ux * e, y0 + uy * e)], color, width)
            t = e + off

    def arrow(self, x0, y0, x1, y1, color, width=2.4, head=13):
        ang = math.atan2(y1 - y0, x1 - x0)
        bx, by = x1 - head * math.cos(ang), y1 - head * math.sin(ang)
        self.line([(x0, y0), (bx, by)], color, width)
        wd = head * 0.46
        p2 = (bx + wd * math.sin(ang), by - wd * math.cos(ang))
        p3 = (bx - wd * math.sin(ang), by + wd * math.cos(ang))
        self.d.polygon([(x1 * S, y1 * S), (p2[0] * S, p2[1] * S), (p3[0] * S, p3[1] * S)], fill=color)

    def chip(self, x, y, s, sz=18, color=INK, bg=WHITE, outline=LINE, pad=16, h=38, bold=False, r=19):
        w = self.tw(s, sz, bold) + pad * 2
        self.box(x, y, x + w, y + h, r=r, fill=bg, outline=outline, width=1.2)
        self.text(x + w / 2, y + h / 2, s, sz, color, "mm", bold)
        return w

    def save(self, path, scale=EXPORT_SCALE):
        self.im.convert("RGB").resize((int(self.w * scale), int(self.h * scale)), Image.LANCZOS).save(path, optimize=True)
        if self.fits:
            print("  [shrink]", self.fits)
        print("saved", os.path.basename(path), f"{int(self.w*scale)}x{int(self.h*scale)}")


# ======================================================================
# 图 1  技术全景
# ======================================================================
def fig1(path):
    W, H = 2400, 1340
    L = Lay(W, H)

    L.text(56, 44, "自动驾驶定位：四类原理，一套融合", 44, INK, "la", True)
    L.text(56, 110, "绝对定位给锚点 · 相对定位保连续 · 环境匹配做精校 · 融合算法出置信度", 22, MUTED)

    cols = [
        dict(name="绝对定位", tag="全局锚点", note="直接给出地球坐标", accent=BLUE, tint=BLUE_T, light=BLUE_L,
             items=[
                 ("GNSS 星座", "单点米级", [
                     "GPS / 北斗 / GLONASS / Galileo",
                     "单点定位 1–5 m，全球全天候可用",
                     "隧道、城市峡谷：遮挡即失锁"]),
                 ("GNSS 增强 RTK / PPP", "厘米~分米级", [
                     "RTK：基站差分，载波相位固定解",
                     "PPP：无需基站，需分钟级收敛",
                     "多路径与遮挡会让固定解直接掉"]),
                 ("蜂窝与车联网定位", "拒止可用", [
                     "5G PRS：多基站测距 / 测角",
                     "C-V2X：与 RSU、他车协同 → 车道级",
                     "依赖覆盖，定位为补盲而非主用"]),
             ]),
        dict(name="相对定位", tag="短时连续", note="不依赖任何外部设施", accent=GREEN, tint=GREEN_T,
             light=GREEN_L,
             items=[
                 ("惯性导航 INS / IMU", "100–200 Hz", [
                     "加速度计 + 陀螺，高频输出姿态",
                     "GNSS 丢失瞬间的「接力棒」",
                     "误差随时间漂移，不能单独长跑"]),
                 ("航位推算 DR", "低成本", [
                     "轮速 + 方向盘转角 + 运动学模型",
                     "量产车最便宜的补位手段",
                     "打滑 / 磨损 / 胎压变化 → 标定漂移"]),
                 ("UWB / 地磁", "场景受限", [
                     "UWB：厘米级，需预布锚点",
                     "车库、园区、AVP 代客泊车",
                     "地磁：无累积误差，需先建磁地图"]),
             ]),
        dict(name="环境特征匹配", tag="局部精校", note="把实时感知与地图对齐", accent=ORANGE,
             tint=ORANGE_T, light=ORANGE_L,
             items=[
                 ("LiDAR 定位", "厘米级", [
                     "实时点云对齐高精点云地图",
                     "ICP：初值好时最准，算力吃紧",
                     "NDT：概率化表达，更快更耐初值"]),
                 ("视觉定位", "语义更稳", [
                     "VO / VIO：跟踪特征点或光流",
                     "语义定位：车道线、标志、灯杆",
                     "不靠低层纹理 → 跨季节更耐用"]),
                 ("高精地图匹配", "三层信息", [
                     "几何层：3D 点云、车道线",
                     "语义层：标志含义、锚点",
                     "拓扑层：车道连接与通行关系"]),
             ]),
        dict(name="传感器融合", tag="统一位姿", note="噪声与不确定性的最优组合", accent=TEAL, tint=TEAL_T,
             light=TEAL_L,
             items=[
                 ("卡尔曼滤波家族", "实时", [
                     "EKF / UKF / ESKF 递推估计",
                     "毫秒级，可直接上车跑",
                     "非线性强时，线性化误差会累积"]),
                 ("因子图优化", "全局最优", [
                     "把各传感器测量建成「因子」",
                     "GNSS / IMU / 视觉 紧耦合重估",
                     "计算量大，多用于后端与离线"]),
                 ("分层融合架构", "工程常态", [
                     "底层：IMU 毫秒级姿态传播",
                     "中层：视觉 / 激光里程计短约束",
                     "高层：GNSS / 地图匹配周期校准"]),
             ]),
    ]

    CW, GAP = 540, 40
    x0 = 56
    hy0, hy1 = 172, 262
    cy0, CH, CSTEP = 322, 252, 278

    for ci, c in enumerate(cols):
        cx = x0 + ci * (CW + GAP)
        cx1 = cx + CW
        a, t = c["accent"], c["tint"]

        # 列头
        L.box(cx, hy0, cx1, hy1, 18, a, a, 0)
        L.text(cx + 26, (hy0 + hy1) / 2, c["name"], 30, WHITE, "lm", True)
        L.text(cx1 - 26, (hy0 + hy1) / 2, c["tag"], 18, (255, 255, 255), "rm")

        # 列注
        ny = hy1 + 16
        L.text(cx + 2, ny, c["note"], 18, MUTED)
        dw = L.tw(c["note"], 18)
        if cx + 2 + dw + 20 < cx1:
            L.dash(cx + 2 + dw + 12, ny + 9, cx1, ny + 9, (203, 213, 225), 1.2, 5, 5)

        for ii, (title, tag, body) in enumerate(c["items"]):
            y0 = cy0 + ii * CSTEP
            y1 = y0 + CH
            L.box(cx, y0, cx1, y1, 16, WHITE, LINE, 1.6)
            L.line([(cx + 4, y0 + 22), (cx + 4, y1 - 22)], a, 5)
            L.text(cx + 26, y0 + 20, title, 25, INK, "la", True, maxw=380)
            L.text(cx + 26, y0 + 58, tag, 17, a, "la", True)
            L.text(cx1 - 22, y0 + 22, f"{ii + 1:02d}", 17, c["light"], "ra", True)
            L.line([(cx + 26, y0 + 88), (cx1 - 22, y0 + 88)], HAIR, 1.4)

            by = y0 + 108
            for b in body:
                L.d.ellipse([(cx + 28) * S, (by + 9) * S, (cx + 34) * S, (by + 15) * S], fill=a)
                L.text(cx + 44, by, b, 19.5, SLATE, maxw=CW - 70)
                by += 36

    # ---------------- 结论 + 场景选型条 ----------------
    by0 = cy0 + 3 * CSTEP + 2
    L.box(56, by0, 2344, by0 + 112, 18, INK, INK, 0)
    L.text(88, by0 + 22, "没有单一技术是全能的", 25, WHITE, "la", True)
    L.text(88, by0 + 62, "互补性才是全场景精度与可靠性的关键", 18, (148, 163, 184))

    scen = [("高速 / 环路", "GNSS/RTK + 高精地图", BLUE),
            ("城市道路", "视觉语义 + LiDAR", ORANGE),
            ("隧道 / 立交", "IMU / DR 递推", GREEN),
            ("地库 / 园区", "UWB + 点云配准", TEAL)]
    sx0, cellw = 620, (2320 - 620) / 4
    for i, (nm, tech, c) in enumerate(scen):
        x = sx0 + i * cellw
        L.line([(x - 18, by0 + 24), (x - 18, by0 + 88)], (45, 58, 78), 1.4)
        L.d.ellipse([(x + 2) * S, (by0 + 33) * S, (x + 10) * S, (by0 + 41) * S], fill=c)
        L.text(x + 20, by0 + 28, nm, 15.5, (148, 163, 184), maxw=cellw - 60)
        L.text(x + 20, by0 + 56, tech, 18, WHITE, maxw=cellw - 46)

    L.text(56, H - 44, "图 1 · 自动驾驶定位技术分类：四类原理与场景选型", 18, FAINT)
    L.text(2344, H - 44, "nxcstudy_ai · 三维视觉 · 2026-11", 18, FAINT, "ra")
    L.save(path)


# ======================================================================
# 图 2  分层架构
# ======================================================================
def fig2(path):
    W, H = 2400, 1420
    L = Lay(W, H)

    L.text(56, 44, "多传感器融合定位：分层架构与数据流", 44, INK, "la", True)
    L.line([(1900, 96), (1938, 96)], (148, 163, 184), 2.2)
    L.text(1948, 96, "数据流", 17, MUTED, "lm")
    L.dash(2022, 96, 2060, 96, TEAL, 1.8, 7, 5)
    L.text(2070, 96, "自检反馈", 17, MUTED, "lm")
    L.text(56, 110, "自上而下是数据流，自下而上是更新频率：越往下越快，越往上越稳", 22, MUTED)

    LX0, LX1 = 56, 296
    CX0, CX1 = 320, 2160

    def band_label(y0, y1, name, sub, a, t, li):
        L.box(LX0, y0, LX1, y1, 16, t, li, 1.4)
        cy = (y0 + y1) / 2
        L.text(LX0 + 22, cy - 12, name, 24, a, "lm", True, maxw=208)
        L.text(LX0 + 22, cy + 18, sub, 15.5, MUTED, "lm", maxw=208)

    # ---------------- L1 传感器层 ----------------
    b1y0, b1y1 = 196, 376
    band_label(b1y0, b1y1, "传感器层", "原始观测", BLUE, BLUE_T, BLUE_L)
    sensors = [("GNSS / RTK / PPP", "全局锚点 · 1–10 Hz"),
               ("IMU", "高频姿态 · 100–200 Hz"),
               ("轮速 / 转角", "运动信息 · 50 Hz"),
               ("LiDAR", "点云 · 10 Hz"),
               ("Camera", "图像 / 语义 · 10–30 Hz"),
               ("UWB / 5G / V2X", "拒止补盲"),
               ("磁力计", "环境指纹")]
    n = len(sensors)
    cw = 248
    gap = (CX1 - CX0 - n * cw) / (n - 1)
    for i, (s, sub) in enumerate(sensors):
        x = CX0 + i * (cw + gap)
        L.box(x, b1y0 + 46, x + cw, b1y0 + 134, 12, WHITE, LINE, 1.5)
        L.text(x + cw / 2, b1y0 + 70, s, 18, INK, "mm", True, maxw=cw - 24)
        L.text(x + cw / 2, b1y0 + 104, sub, 13.5, MUTED, "mm", maxw=cw - 18)

    # ---------------- L2 前端解算层 ----------------
    b2y0, b2y1 = 414, 706
    band_label(b2y0, b2y1, "前端解算层", "各自先解一遍", GREEN, GREEN_T, GREEN_L)

    groups = [
        (CX0, 872, "绝对解算", BLUE, BLUE_T,
         ["RTK / PPP 固定解", "5G PRS 多基站测距", "C-V2X / UWB 锚点测距"]),
        (892, 1432, "相对递推", GREEN, GREEN_T,
         ["INS / IMU 姿态传播", "航位推算 DR（轮速 + 转角）", "车辆运动学 / 非完整约束"]),
        (1452, CX1, "环境匹配", ORANGE, ORANGE_T,
         ["LiDAR 点云配准 ICP / NDT", "VO / VIO · 语义定位", "高精地图匹配（几何 / 语义 / 拓扑）"]),
    ]
    for gx0, gx1, gname, ga, gt, gitems in groups:
        L.box(gx0, b2y0, gx1, b2y1, 16, WHITE, LINE, 1.6)
        L.text(gx0 + 20, b2y0 + 16, gname, 19, ga, "la", True)
        L.line([(gx0 + 18, b2y0 + 50), (gx1 - 18, b2y0 + 50)], HAIR, 1.4)
        start = b2y0 + 62
        for j, it in enumerate(gitems):
            y = start + j * 66
            L.box(gx0 + 16, y, gx1 - 16, y + 54, 11, gt, gt, 0)
            L.text((gx0 + gx1) / 2, y + 27, it, 18, (30, 41, 59), "mm", maxw=gx1 - gx0 - 56)

    # ---------------- L3 融合优化层 ----------------
    b3y0, b3y1 = 764, 960
    band_label(b3y0, b3y1, "融合优化层", "最优估计 + 自检", TEAL, TEAL_T, TEAL_L)
    fus = [
        (CX0, 992, "ESKF / EKF / UKF", "递推滤波 · 50–100 Hz 实时"),
        (1012, 1652, "因子图优化（后端）", "GNSS / IMU / 视觉 紧耦合 · 全局重估"),
        (1672, CX1, "置信度 / 故障检测", "完好性监测 → 降级决策"),
    ]
    for fx0, fx1, t1, t2 in fus:
        L.box(fx0, b3y0, fx1, b3y1, 16, WHITE, LINE, 1.6)
        cy = (b3y0 + b3y1) / 2
        L.text((fx0 + fx1) / 2, cy - 16, t1, 22, INK, "mm", True, maxw=fx1 - fx0 - 40)
        L.text((fx0 + fx1) / 2, cy + 18, t2, 17, MUTED, "mm", maxw=fx1 - fx0 - 40)

    # ---------------- L4 输出层 ----------------
    b4y0, b4y1 = 1000, 1120
    band_label(b4y0, b4y1, "输出", "带不确定度的位姿", INK, (241, 245, 249), LINE)
    L.box(CX0, b4y0, CX1, b4y1, 16, WHITE, LINE, 1.6)
    L.text(CX0 + 26, b4y0 + 24, "车辆位姿", 24, INK, "la", True)
    L.text(CX0 + 26, b4y0 + 66, "位置 x / y / z · 姿态 roll / pitch / yaw · 速度 · 协方差", 19, SLATE)
    L.chip(CX1 - 24 - 118, b4y0 + 40, "10–100 Hz", 18, INK, (241, 245, 249), LINE, 18, 42, True)

    # ---------------- 层间箭头 ----------------
    AR = (148, 163, 184)
    for x in (596, 1162, 1806):
        L.arrow(x, b1y1 + 6, x, b2y0 - 6, AR, 2.2, 11)
    for x in (676, 1332, 1916):
        L.arrow(x, b2y1 + 6, x, b3y0 - 6, AR, 2.2, 11)
        L.arrow(x, b3y1 + 6, x, b4y0 - 6, AR, 2.2, 11)

    # ---------------- 自检 / 降级 回路 ----------------
    cy1 = (b1y0 + b1y1) / 2
    cy3 = (b3y0 + b3y1) / 2
    rx = 2336
    L.dash(CX1 + 4, cy3, rx, cy3, TEAL, 1.8, 8, 6)
    L.dash(rx, cy3, rx, cy1, TEAL, 1.8, 8, 6)
    L.arrow(rx, cy1, CX1 + 8, cy1, TEAL, 1.8, 11)
    L.vtext(rx - 22, (cy1 + cy3) / 2, "自检回路 · 降级 / 切换源 / 重初始化", 15, TEAL, True)

    # ---------------- 频率梯队 ----------------
    ly0, ly1 = 1160, 1300
    L.box(56, ly0, 2344, ly1, 16, WHITE, LINE, 1.6)
    L.text(88, ly0 + 40, "更新频率梯队", 20, INK, "la", True)
    L.text(88, ly0 + 78, "快的管短时，慢的管全局", 16, MUTED)
    freq = [("IMU", "100–200 Hz", BLUE), ("融合输出", "50–100 Hz", TEAL), ("轮速 / 转角", "50 Hz", GREEN),
            ("Camera", "10–30 Hz", ORANGE), ("LiDAR", "10 Hz", ORANGE), ("GNSS", "1–10 Hz", BLUE)]
    cw2 = (2344 - 40 - 380) / 6
    for i, (nm, v, c) in enumerate(freq):
        x = 420 + i * cw2
        if i:
            L.line([(x - 12, ly0 + 26), (x - 12, ly1 - 26)], HAIR, 1.2)
        L.text(x + 16, ly0 + 44, nm, 16, MUTED)
        L.text(x + 16, ly0 + 78, v, 24, c, "la", True, maxw=cw2 - 40)

    L.text(56, H - 40, "图 2 · 多传感器融合定位的分层数据流 · 实线为数据流，虚线为自检反馈", 18, FAINT)
    L.text(2344, H - 40, "nxcstudy_ai · 三维视觉 · 2026-11", 18, FAINT, "ra")
    L.save(path)


# ======================================================================
# 图 3  一次隧道穿越的时序：断、接、重锁
# ======================================================================
def fig3(path):
    W, H = 2400, 1460
    L = Lay(W, H)

    L.text(56, 40, "32 秒没有 GNSS：定位怎么断、怎么接、怎么重锁", 44, INK, "la", True)
    L.text(56, 104, "同一条时间轴上四类技术轮流供数；误差曲线是它们共同的结果，不是某一家的功劳", 21, MUTED)

    PX0, PX1 = 300, 2280
    PY0, PY1 = 276, 932
    TMAX = 60.0

    def tx(t):
        return PX0 + t / TMAX * (PX1 - PX0)

    def ty(e):
        return PY1 - (math.log10(max(e, 1e-3)) + 2) / 3.0 * (PY1 - PY0)

    # ---- 误差模型：E = 无外部约束时的惯导/DR 误差；real = 加了地图匹配之后 ----
    def E(t):
        if t <= 8:
            return 0.02
        if t <= 13:
            return 0.02 + 0.06 * ((t - 8) / 5.0) ** 1.6
        return 0.08 + 0.0017 * (t - 13) ** 2

    def real(t):
        if t <= 22:
            return E(t) + 0.0012 * math.sin(t * 6.1) + 0.0008 * math.sin(t * 2.3)
        if t < 40:
            e = E(t) * math.exp(-0.12 * (t - 22))
            e += 0.020 * ((t - 22) % 1.2) / 1.2 * math.exp(-(t - 22) / 40)
            return e + 0.0008 * math.sin(t * 5.7)
        return 0.02 + 0.125 * math.exp(-(t - 40) / 1.6) + 0.0010 * math.sin(t * 5.3)

    def fill_poly(pts, color):
        ov = Image.new("RGBA", L.im.size, (0, 0, 0, 0))
        ImageDraw.Draw(ov).polygon([(x * S, y * S) for x, y in pts], fill=color)
        L.im.alpha_composite(ov)

    # ---- 拒止区间底色 ----
    fill_poly([(tx(8), PY0), (tx(40), PY0), (tx(40), PY1), (tx(8), PY1)], (15, 23, 42, 10))
    L.text(tx(8) + 11, PY0 + 10, "GNSS 拒止区间 · 32 s", 15, FAINT)

    # ---- 网格 ----
    for e, lab in [(0.01, "1 cm"), (0.1, "10 cm"), (10, "10 m")]:
        y = ty(e)
        L.line([(PX0, y), (PX1, y)], LINE, 1.2)
        L.text(PX0 - 12, y, lab, 15, FAINT, "rm", False)
    for t in range(0, 61, 10):
        x = tx(t)
        L.line([(x, PY0), (x, PY1)], (238, 242, 247), 1.0)
        L.text(x, PY1 + 12, f"{t}s", 15, FAINT, "ma")

    # ---- 阈值线 ----
    L.dash(PX0, ty(1.0), PX1, ty(1.0), (220, 38, 38), 1.8, 10, 7)
    L.text(PX1 - 6, ty(1.0) - 26, "1 m · 定位失效线", 15, (185, 28, 28), "ra", True)
    L.dash(PX0, ty(0.3), PX1, ty(0.3), (217, 119, 6), 1.6, 8, 7)
    L.text(PX1 - 6, ty(0.3) - 24, "0.3 m · 车道级可用上限", 15, (180, 83, 9), "ra", True)

    # ---- 曲线 ----
    ts = [i * 0.2 for i in range(301)]
    fill_poly([(tx(t), ty(real(t))) for t in ts] + [(PX1, PY1), (PX0, PY1)], (13, 137, 128, 16))
    gp = [(tx(t), ty(E(t))) for t in ts if t >= 22]
    L.line([(gp[i][0], gp[i][1]) for i in range(len(gp) - 1)], (148, 163, 184), 2.6)
    L.line(gp, (148, 163, 184), 2.6)
    L.line([(tx(t), ty(real(t))) for t in ts], INK, 3.2)

    # ---- 事件点 ----
    for t in (8, 13, 22, 40):
        x, y = tx(t), ty(real(t))
        L.d.ellipse([(x - 8) * S, (y - 8) * S, (x + 8) * S, (y + 8) * S], fill=WHITE, outline=INK, width=int(2.2 * S))

    # ① 失锁
    L.text(tx(8) - 12, ty(0.055), "GNSS 失锁", 15, BLUE, "ra", True)
    L.line([(tx(8) - 10, ty(0.048)), (tx(8) - 3, ty(0.026))], BLUE, 1.2)
    # ② 地图匹配介入
    L.text(tx(22) - 18, ty(0.62), "地图匹配介入", 15, ORANGE, "ra", True)
    L.text(tx(22) - 18, ty(0.47), "误差被按回分米级", 14, MUTED, "ra")
    L.line([(tx(22) - 14, ty(0.44)), (tx(22) - 5, ty(0.245))], ORANGE, 1.2)
    # ③ 纯惯导撞线
    L.text(tx(36.3) + 16, ty(1.0) + 24, "没有地图匹配，36 s 撞线", 15, (185, 28, 28), "la", True)
    L.line([(tx(36.3) + 12, ty(1.0) + 20), (tx(36.3) + 4, ty(1.0) + 5)], (185, 28, 28), 1.2)
    # ④ 重固定
    L.text(tx(40) + 14, ty(0.215), "出洞 · 重固定", 15, BLUE, "la", True)

    # ---- 图例 ----
    lx = PX1 - 4
    for label, col, kind in [("只有惯导（假设）", (148, 163, 184), "dash"), ("融合后误差", INK, "solid")]:
        w = L.tw(label, 15)
        L.text(lx, PY0 + 20, label, 15, MUTED, "ra")
        x2 = lx - w - 12
        if kind == "dash":
            L.dash(x2 - 34, PY0 + 28, x2, PY0 + 28, col, 2.4, 7, 5)
        else:
            L.line([(x2 - 34, PY0 + 28), (x2, PY0 + 28)], col, 2.8)
        lx = x2 - 34 - 26

    # ---- 相位带 ----
    phases = [(0, 8, "①", "GNSS 固定解", "厘米级 · 置信度高", BLUE, BLUE_T),
              (8, 13, "②", "IMU 单独接力", "5 s，误差 < 10 cm", GREEN, GREEN_T),
              (13, 22, "③", "轮速 / DR 递推", "9 s，缓坡上行", GREEN, GREEN_T),
              (22, 40, "④", "点云 / 语义匹配", "18 s，压回分米级", ORANGE, ORANGE_T),
              (40, 60, "⑤", "GNSS 重捕获", "2 s 重新固定", BLUE, BLUE_T)]
    by0, by1 = 168, 244
    for t0, t1, num, name, sub, c, tint in phases:
        x0, x1 = tx(t0) + 1, tx(t1) - 1
        L.box(x0, by0, x1, by1, 10, tint, c, 1.4)
        L.d.rectangle([(x0 + 8) * S, by0 * S + 1, (x1 - 8) * S, (by0 + 6) * S], fill=c)
        L.text(x0 + 16, by0 + 18, f"{num} {name}", 16, c, "la", True, maxw=x1 - x0 - 28)
        L.text(x0 + 16, by0 + 47, sub, 13, MUTED, maxw=x1 - x0 - 24)

    # ---- 轴 ----
    L.line([(PX0, PY0 - 10), (PX0, PY1)], LINE, 1.6)
    L.line([(PX0, PY1), (PX1, PY1)], LINE, 1.6)
    L.vtext(74, (PY0 + PY1) / 2, "定位误差（对数轴）", 16, MUTED, True)

    # ---- 接力甘特 ----
    L.text(56, 962, "技术接力（同一时间轴）", 15, MUTED, "la", True)
    rows = [
        ("GNSS / RTK", BLUE, BLUE_T, [(0, 8, "固定解", 0), (8, 40, "失锁 / 拒止", 1), (40, 60, "重捕获", 0)]),
        ("IMU", GREEN, GREEN_T, [(0, 60, "连续输出 100–200 Hz", 0)]),
        ("轮速 / DR", GREEN, GREEN_T, [(13, 40, "里程计递推", 0)]),
        ("LiDAR / 视觉匹配", ORANGE, ORANGE_T, [(22, 40, "点云 / 语义约束", 0)]),
        ("融合输出", TEAL, TEAL_T, [(0, 60, "位姿 + 协方差", 0)]),
    ]
    ry = 990
    for name, c, tint, segs in rows:
        L.d.ellipse([(58) * S, (ry + 11) * S, (66) * S, (ry + 19) * S], fill=c)
        L.text(76, ry + 6, name, 16, INK, "la", True)
        for t0, t1, lab, lost in segs:
            x0, x1 = tx(t0) + 2, tx(t1) - 2
            if lost:
                L.box(x0, ry, x1, ry + 30, 8, (250, 251, 253), (203, 213, 225), 1.2)
            else:
                L.box(x0, ry, x1, ry + 30, 8, tint, c, 1.2)
            L.text((x0 + x1) / 2, ry + 15, lab, 14, MUTED if lost else c, "mm", not lost,
                   maxw=x1 - x0 - 20)
        ry += 42

    # ---- 三条结论 ----
    cards = [("① 为什么前 5 秒可以不管",
              ["IMU 200 Hz 输出，5 s 内位置偏差仍在 10 cm 量级；", "远小于 3.5 m 的车道宽度，够用。"]),
             ("② 13–22 s 是分水岭",
              ["纯惯导误差随时间的平方增长，轮速再叠加打滑；", "误差从分米滑向米，必须引入外部约束。"]),
             ("③ 重锁不是切换，是融合",
              ["地图匹配没有「接管」IMU，只是把权重调大；", "所以曲线是平滑回落，不是跳变。"])]
    cy = 1240
    for i, (title, lines) in enumerate(cards):
        x0 = 56 + i * 770
        L.box(x0, cy, x0 + 740, cy + 122, 14, WHITE, LINE, 1.6)
        L.line([(x0 + 4, cy + 20), (x0 + 4, cy + 102)], TEAL, 5)
        L.text(x0 + 24, cy + 16, title, 18, INK, "la", True)
        for j, ln in enumerate(lines):
            L.text(x0 + 24, cy + 52 + j * 28, ln, 15, SLATE, maxw=690)

    L.text(56, H - 40, "图 3 · 一次隧道穿越的定位时序：断点、接力与重锁", 18, FAINT)
    L.text(2344, H - 40, "nxcstudy_ai · 三维视觉 · 2026-11", 18, FAINT, "ra")
    L.save(path)


# -*- coding: utf-8 -*-
# 图 4–9，追加到 _make_localization.py


def tbl(L, x0, y0, widths, head, rows, hh=48, rh=54, fs=15.5):
    total = sum(widths)
    L.d.rectangle([x0 * S, y0 * S, (x0 + total) * S, (y0 + hh) * S], fill=INK)
    cx = x0
    for i, w in enumerate(widths):
        L.text(cx + 18, y0 + hh / 2, head[i], fs + 0.5, WHITE, "lm", True)
        cx += w
    y = y0 + hh
    for j, row in enumerate(rows):
        L.d.rectangle([x0 * S, y * S, (x0 + total) * S, (y + rh) * S],
                      fill=WHITE if j % 2 == 0 else (250, 251, 253))
        L.line([(x0, y), (x0 + total, y)], LINE, 1.2)
        cx = x0
        for i, w in enumerate(widths):
            L.text(cx + 18, y + rh / 2, row[i], fs, SLATE if i else INK, "lm", i == 0, maxw=w - 30)
            cx += w
            if i:
                L.line([(cx - 0, y + 8), (cx, y + rh - 8)], (238, 242, 247), 1)
        y += rh
    L.line([(x0, y), (x0 + total, y)], LINE, 1.4)


# ======================================================================
# 图 4  坐标系与变换链
# ======================================================================
def fig4(path):
    W, H = 2400, 1090
    L = Lay(W, H)
    L.text(56, 40, "坐标系与变换链：一个点从卫星走到像素", 44, INK, "la", True)
    L.text(56, 104, "T_AB 只做一件事：把 A 系下的坐标变成 B 系下的坐标 —— p_B = T_AB · p_A；顺序不可交换", 21, MUTED)

    nodes = [("世界系", "WGS84 经纬高 / UTM 米"),
             ("局部 ENU", "切平面，单位米，原点 = 起点"),
             ("车体系", "后轴中心，x 前 y 左 z 上"),
             ("传感器系", "IMU / LiDAR / Camera 各自原点"),
             ("像素系", "u 向右，v 向下，单位 pixel")]
    trans = [("T_enu_wgs", "一次性"), ("T_car_enu", "每帧更新"),
             ("T_sensor_car", "出厂标定"), ("K·[R|t]", "投影")]

    NW, NH, GAP = 356, 118, 125
    ny0 = 186
    for i, (nm, sub) in enumerate(nodes):
        x = 60 + i * (NW + GAP)
        L.box(x, ny0, x + NW, ny0 + NH, 14, WHITE, LINE, 1.6)
        L.text(x + 18, ny0 + 14, nm, 19, INK, "la", True)
        L.text(x + 18, ny0 + 46, sub, 13.5, MUTED, maxw=NW - 36)
        tx0, ty0 = x + NW - 58, ny0 + NH - 24
        if i == 4:
            L.line([(tx0, ty0), (tx0 + 34, ty0)], (37, 99, 235), 1.6)
            L.line([(tx0, ty0), (tx0, ty0 - 32)], (220, 38, 38), 1.6)
            L.text(tx0 + 38, ty0 - 8, "u", 11, (37, 99, 235))
            L.text(tx0 - 6, ty0 - 48, "v", 11, (220, 38, 38))
        else:
            L.arrow(tx0, ty0, tx0 + 30, ty0, (220, 38, 38), 1.5, 7)
            L.arrow(tx0, ty0, tx0 - 19, ty0 - 19, (22, 163, 74), 1.5, 7)
            L.arrow(tx0, ty0, tx0 - 19, ty0 + 19, (37, 99, 235), 1.5, 7)

    for i, (tn, note) in enumerate(trans):
        xa = 60 + i * (NW + GAP) + NW + 6
        xb = 60 + (i + 1) * (NW + GAP) - 6
        ym = ny0 + 48
        L.arrow(xa, ym, xb, ym, (148, 163, 184), 2.0, 10)
        cx = (xa + xb) / 2
        w = L.tw(tn, 13.5, True) + 16
        L.box(cx - w / 2, ym - 32, cx + w / 2, ym - 6, 13, WHITE, BLUE_L, 1.2)
        L.text(cx, ym - 19, tn, 13.5, BLUE, "mm", True)
        L.text(cx, ym + 10, note, 12.5, MUTED, "ma")

    # ---- 一个点走一遍 ----
    L.text(56, 344, "同一个车道线角点，在四个系里的四张脸", 20, INK, "la", True)
    cards = [("来源", "世界系 · WGS84", ["31.2304° N", "121.4737° E", "4.2 m"], "GNSS 接收机直接给出的就是它", BLUE, BLUE_T),
             ("T_enu_wgs", "局部 ENU", ["x    12.4 m", "y   -35.8 m", "z     0.2 m"], "减原点，再投影到切平面", BLUE, BLUE_T),
             ("T_car_enu", "车体系", ["x     8.10 m", "y     1.75 m", "z     0.10 m"], "定位模块对外的输出（后轴中心）", GREEN, GREEN_T),
             ("K·[R|t]", "像素系", ["u   640 px", "v   380 px", ""], "内参 + 外参一起投到图像", ORANGE, ORANGE_T)]
    for i, (strip, name, vals, note, c, t) in enumerate(cards):
        x = 60 + i * 580
        L.box(x, 376, x + 540, 604, 14, WHITE, LINE, 1.6)
        L.box(x, 376, x + 540, 406, 14, c, c, 0)
        L.d.rectangle([(x + 2) * S, (400) * S, (x + 538) * S, (406) * S], fill=c)
        L.text(x + 270, 391, strip, 13, WHITE, "mm", True)
        L.text(x + 22, 422, name, 17, INK, "la", True)
        for j, v in enumerate(vals):
            if v:
                L.text(x + 22, 456 + j * 34, v, 19.5, c, "la", True)
        L.text(x + 22, 570, note, 13.5, MUTED, maxw=496)

    # ---- 三个坑 ----
    pits = [("① 方向：谁变谁", ["T_AB 把 A 系的点变到 B 系：p_B = T_AB · p_A。",
                            "反过来要整体求逆 —— 旋转转置了，平移也得跟着换。",
                            "自检：把车头正前方 10 m 的点送进去，看结果还是不是 10 m。"]),
            ("② 顺序：左乘不能换", ["p_pix = K · T_cam_car · T_car_enu · p_enu",
                               "右乘或换序 = 换了一个物理含义，离原点越远错得越离谱。",
                               "写法：从最外层（像素 / 地图）往里乘，一层层剥回世界系。"]),
            ("③ 三个隐形偏移", ["外参方向（谁到谁）、杆臂（天线不在后轴中心）、",
                            "时间戳（三个传感器不能各说各话），缺一个都对齐不上。",
                            "顺序：先对时间戳，再补杆臂，最后才是外参。"]), ]
    for i, (title, body) in enumerate(pits):
        x = 60 + i * 740
        L.box(x, 636, x + 740, 820, 14, WHITE, LINE, 1.6)
        L.line([(x + 4, 656), (x + 4, 800)], (220, 38, 38), 4)
        L.text(x + 24, 654, title, 18, INK, "la", True)
        for j, b in enumerate(body):
            L.text(x + 24, 690 + j * 33, b, 14.5, SLATE, maxw=696)

    # ---- 自检三问 ----
    L.box(56, 852, 2344, 962, 18, INK, INK, 0)
    L.text(88, 890, "拿到一个位姿，先问三句", 24, WHITE, "la", True)
    for i, q in enumerate(["① 是哪个系到哪个系的位姿？", "② 它带时间戳吗？", "③ 协方差是谁的协方差？"]):
        L.text(700 + i * 560, 900, q, 17, (203, 213, 225))

    L.text(56, H - 42, "图 4 · 坐标系与变换链：世界系 → 局部 ENU → 车体系 → 传感器系 → 像素", 18, FAINT)
    L.text(2344, H - 42, "nxcstudy_ai · 三维视觉 · 2026-11", 18, FAINT, "ra")
    L.save(path)


# ======================================================================
# 图 5  递推滤波 vs 因子图
# ======================================================================
def fig5(path):
    W, H = 2400, 1160
    L = Lay(W, H)
    L.text(56, 40, "同一组测量，两种解法：递推滤波 vs 因子图", 44, INK, "la", True)
    L.text(56, 104, "左边每来一帧就更新一次，只留当前状态；右边把约束攒成一张图，一起解", 21, MUTED)

    # ---------------- 左：递推 ----------------
    L.text(56, 168, "递推滤波 · EKF / ESKF", 24, BLUE, "la", True)
    L.text(56, 202, "每帧一次「预测 + 更新」，状态只有一份", 15, MUTED)

    def fbox(x0, y0, x1, y1, t1, t2, c):
        L.box(x0, y0, x1, y1, 14, WHITE, c, 1.8)
        cy = (y0 + y1) / 2
        L.text((x0 + x1) / 2, cy - 14, t1, 18, INK, "mm", True, maxw=x1 - x0 - 30)
        L.text((x0 + x1) / 2, cy + 16, t2, 13, MUTED, "mm", maxw=x1 - x0 - 26)

    fbox(56, 244, 466, 340, "上一时刻", "状态 x，协方差 P", BLUE)
    fbox(746, 244, 1156, 340, "预测：IMU 积分", "x = f(x, u)，P = F·P·F^T + Q", BLUE)
    fbox(746, 408, 1156, 504, "观测更新", "K = P·H^T·(H·P·H^T + R)^-1", BLUE)
    fbox(56, 408, 466, 504, "后验 / 输出", "x ← x + K·(z - h(x))", BLUE)

    AR = (148, 163, 184)
    L.arrow(470, 292, 742, 292, AR, 2.2, 11)
    L.text(606, 276, "IMU", 13, MUTED, "ma")
    L.arrow(951, 344, 951, 404, AR, 2.2, 11)
    L.text(965, 374, "GNSS / 视觉", 13, MUTED, "lm")
    L.arrow(742, 456, 470, 456, AR, 2.2, 11)
    L.arrow(261, 404, 261, 344, AR, 2.2, 11)
    L.text(272, 374, "t+1", 13, MUTED, "lm")

    L.text(56, 540, "只有一份状态，所以「回改历史」不可能：一次线性化错了，误差就留在里面。", 14, MUTED, maxw=1100)

    # ---------------- 右：因子图 ----------------
    L.text(1250, 168, "因子图 · 滑窗 / 全局优化", 24, TEAL, "la", True)
    L.text(1250, 202, "所有时刻一起解，非线性可以反复线性化", 15, MUTED)

    xs = [1300, 1500, 1700, 1900, 2100]
    ny = 400
    for i, x in enumerate(xs):
        L.d.ellipse([(x - 26) * S, (ny - 26) * S, (x + 26) * S, (ny + 26) * S], fill=WHITE, outline=INK, width=int(2 * S))
        L.text(x, ny, f"x{i}", 15, INK, "mm", True)
    for i in range(4):
        L.line([(xs[i] + 26, ny), (xs[i + 1] - 26, ny)], INK, 2)

    def fac(x, y, c, s=13):
        L.box(x - s, y - s, x + s, y + s, 5, c, c, 0)
        L.line([(x + s, ny), (x + s, y + s)], c, 1.4) if y < ny else L.line([(x, y - s), (x, ny + 26)], c, 1.4)

    for i in range(4):
        mx = (xs[i] + xs[i + 1]) / 2
        L.line([(mx, ny - 26), (mx, ny - 62)], GREEN, 1.6)
        L.box(mx - 13, ny - 75, mx + 13, ny - 49, 5, GREEN, GREEN, 0)
    # 先验
    L.line([(xs[0], ny - 26), (xs[0], ny - 62)], FAINT, 1.6)
    L.box(xs[0] - 13, ny - 75, xs[0] + 13, ny - 49, 5, (203, 213, 225), (203, 213, 225), 0)
    # GNSS / 视觉
    for x, c in [(xs[1], BLUE), (xs[3], BLUE), (xs[2], ORANGE), (xs[4], ORANGE)]:
        L.line([(x, ny + 26), (x, ny + 62)], c, 1.6)
        L.box(x - 13, ny + 49, x + 13, ny + 75, 5, c, c, 0)
    # 回环
    L.dash(xs[4], ny - 30, xs[4], ny - 150, (220, 38, 38), 1.6, 7, 5)
    L.dash(xs[4], ny - 150, xs[0], ny - 150, (220, 38, 38), 1.6, 7, 5)
    L.dash(xs[0], ny - 150, xs[0], ny - 30, (220, 38, 38), 1.6, 7, 5)
    L.box(1700 - 13, ny - 163, 1700 + 13, ny - 137, 5, (220, 38, 38), (220, 38, 38), 0)
    L.text(1722, ny - 176, "回环：直接加一条边，历史就被改了", 13, (185, 28, 28), "lm")

    # 图例
    lg = [("IMU 预积分", GREEN), ("GNSS 观测", BLUE), ("视觉观测", ORANGE), ("先验", (203, 213, 225))]
    lx = 1250
    for name, c in lg:
        L.box(lx, ny + 112, lx + 20, ny + 132, 4, c, c, 0)
        L.text(lx + 28, ny + 122, name, 13.5, MUTED, "lm")
        lx += 28 + L.tw(name, 13.5) + 34

    L.text(1250, 576, "min Σ ||r_i||^2 —— 一次解一大片；非线性强、有回环时明显更准。", 14, MUTED, maxw=1080)

    # ---------------- 对比表 ----------------
    rows = [["状态表达", "当前状态 + 协方差，马尔可夫假设", "窗口内所有状态与约束，一起解"],
            ["非线性", "每步线性化一次，误差会累积", "可反复线性化，逼近真正的极小点"],
            ["回环 / 重访", "几乎无法回头改历史", "天然支持，加一条边就行"],
            ["计算", "每帧开销固定，实时友好", "随窗口增长，要滑窗 / 边缘化"],
            ["典型落地", "车端组合导航、实时定位", "离线建图、后处理、地图与标定优化"]]
    tbl(L, 56, 640, [320, 960, 1000], ["维度", "递推滤波（EKF / ESKF）", "因子图（滑窗 / 全局）"], rows)

    L.box(56, 1000, 2344, 1082, 16, (241, 245, 249), LINE, 1.4)
    L.text(88, 1026, "一句话：车端要快，选递推；离线要准，选因子图。ESKF 也可以看成「只有两帧的因子图」。",
           20, INK, "la", True)

    L.text(56, H - 40, "图 5 · 递推滤波与因子图：同一组测量，两种解法", 18, FAINT)
    L.text(2344, H - 40, "nxcstudy_ai · 三维视觉 · 2026-11", 18, FAINT, "ra")
    L.save(path)


# ======================================================================
# 图 6  松耦合 vs 紧耦合
# ======================================================================
def fig6(path):
    W, H = 2400, 1120
    L = Lay(W, H)
    L.text(56, 40, "松耦合与紧耦合：GNSS 是在解算之后，还是解算之中进滤波器", 42, INK, "la", True)
    L.text(56, 104, "同一套硬件，差别只在「把哪一层数据交给融合」—— 但失效行为完全不同", 21, MUTED)

    def pbox(x0, y0, x1, y1, t1, t2, c, t):
        L.box(x0, y0, x1, y1, 13, WHITE, c, 1.7)
        cy = (y0 + y1) / 2
        L.text((x0 + x1) / 2, cy - 13 if t2 else cy, t1, 17, INK, "mm", True, maxw=x1 - x0 - 26)
        if t2:
            L.text((x0 + x1) / 2, cy + 15, t2, 12.5, MUTED, "mm", maxw=x1 - x0 - 26)

    # ---------------- 松耦合 ----------------
    L.box(56, 168, 2344, 486, 18, (247, 250, 253), LINE, 1.5)
    L.text(84, 190, "松耦合", 26, BLUE, "la", True)
    L.text(84, 230, "先解算，再融合", 15, MUTED)
    L.text(84, 258, "PVT 是输入", 13, FAINT)

    pbox(330, 300, 590, 396, "GNSS 接收机", "原始观测量", BLUE, BLUE_T)
    pbox(640, 300, 940, 396, "PVT 解算", "单点 / RTK / PPP", BLUE, BLUE_T)
    pbox(1840, 296, 2180, 400, "EKF / ESKF", "融合滤波器", BLUE, BLUE_T)
    pbox(1560, 192, 2160, 262, "视觉 / 激光里程计", "位姿增量", TEAL, TEAL_T)
    pbox(1560, 430, 2160, 486, "IMU", "预积分", GREEN, GREEN_T)

    AR = (148, 163, 184)
    L.arrow(594, 348, 636, 348, AR, 2.2, 10)
    L.arrow(944, 348, 1836, 348, AR, 2.2, 10)
    L.text(1390, 330, "经纬度 + 速度（已经解算好了）", 15, INK, "ma", True)
    L.text(1390, 376, "丢一颗星？整段 PVT 一起丢。", 13.5, FAINT, "ma")
    L.arrow(2000, 266, 2000, 292, AR, 2.0, 9)
    L.arrow(2000, 426, 2000, 404, AR, 2.0, 9)

    # ---------------- 紧耦合 ----------------
    L.box(56, 508, 2344, 826, 18, (252, 250, 247), LINE, 1.5)
    L.text(84, 530, "紧耦合", 26, ORANGE, "la", True)
    L.text(84, 570, "边观测边融合", 15, MUTED)
    L.text(84, 598, "原始观测进滤波器", 13, FAINT)

    pbox(330, 640, 590, 736, "GNSS 接收机", "原始观测量", ORANGE, ORANGE_T)
    pbox(640, 640, 940, 736, "原始观测", "伪距 / 多普勒 / 载波", ORANGE, ORANGE_T)
    pbox(1840, 636, 2180, 740, "EKF / 因子图", "与 IMU 联合估计", ORANGE, ORANGE_T)
    pbox(1560, 532, 2160, 602, "视觉 / 激光里程计", "特征 / 位姿增量", TEAL, TEAL_T)
    pbox(1560, 770, 2160, 826, "IMU", "预积分", GREEN, GREEN_T)

    L.arrow(594, 688, 636, 688, AR, 2.2, 10)
    L.arrow(944, 688, 1836, 688, AR, 2.2, 10)
    L.text(1390, 670, "每颗卫星一条观测方程，直接进滤波器", 15, INK, "ma", True)
    L.text(1390, 716, "丢一颗星？只丢那一条方程。", 13.5, FAINT, "ma")
    L.arrow(2000, 606, 2000, 632, AR, 2.0, 9)
    L.arrow(2000, 766, 2000, 744, AR, 2.0, 9)

    # 被绕过的 PVT
    L.dash(640, 532, 930, 532, (203, 213, 225), 1.5, 6, 5)
    L.dash(640, 600, 930, 600, (203, 213, 225), 1.5, 6, 5)
    L.dash(640, 532, 640, 600, (203, 213, 225), 1.5, 6, 5)
    L.dash(930, 532, 930, 600, (203, 213, 225), 1.5, 6, 5)
    L.text(785, 552, "PVT 解算", 15, (148, 163, 184), "mm")
    L.text(785, 578, "仍在接收机内部，但不进融合", 11.5, (148, 163, 184), "mm")
    L.line([(946, 552), (964, 570)], (220, 38, 38), 2.2)
    L.line([(964, 552), (946, 570)], (220, 38, 38), 2.2)
    L.text(974, 562, "这一层被跳过", 12.5, (185, 28, 28), "lm", True)

    # ---------------- 三条差别 ----------------
    diffs = [("卫星少的时候", ["松耦合要 ≥4 颗才解得出 PVT，凑不够就整段丢；", "紧耦合 2–3 颗也能给约束，只是权重变小。"]),
             ("失效行为", ["松耦合是全有或全无，切进去的瞬间会跳一下；", "紧耦合逐颗降权，输出连续、也更容易解释。"]),
             ("代价", ["要拿到原始观测（接收机协议 / 授权），还要做天线相位中心、", "杆臂和时间对齐 —— 工程量和调试难度都上一个台阶。"])]
    for i, (title, body) in enumerate(diffs):
        x = 60 + i * 740
        L.box(x, 872, x + 740, 1022, 14, WHITE, LINE, 1.6)
        L.text(x + 24, 894, title, 18, INK, "la", True)
        for j, b in enumerate(body):
            L.text(x + 24, 934 + j * 32, b, 14, SLATE, maxw=696)

    L.text(56, H - 40, "图 6 · 松耦合与紧耦合：GNSS 数据在哪一层进入融合", 18, FAINT)
    L.text(2344, H - 40, "nxcstudy_ai · 三维视觉 · 2026-11", 18, FAINT, "ra")
    L.save(path)
# -*- coding: utf-8 -*-
# 图 7–9


# ======================================================================
# 图 7  高精地图三层
# ======================================================================
def fig7(path):
    W, H = 2400, 1120
    L = Lay(W, H)
    L.text(56, 40, "高精地图的三层：几何、语义、拓扑", 44, INK, "la", True)
    L.text(56, 104, "存的东西不同，匹配时也从粗到细用 —— 先用拓扑筛，再用语义对，最后用几何压", 21, MUTED)

    bands = [(196, "拓扑层", "车道之间怎么连，能不能变道", TEAL, TEAL_T,
              ["存：车道中心线、连接关系、通行方向", "用：先筛候选，把 200 条降到 5 条"]),
             (426, "语义层", "这条道限速多少，这个标志什么意思", ORANGE, ORANGE_T,
              ["存：限速、停止线、路牌、路口属性", "用：找语义锚点，先粗对齐"]),
             (656, "几何层", "墙在哪、车道线在哪，厘米级", BLUE, BLUE_T,
              ["存：点云、车道线、路沿、高程", "用：ICP / 配准，把误差压到厘米"])]

    stats = {"拓扑层": ("几十 KB / km", "几乎不变"),
             "语义层": ("几百 KB / km", "季度更新"),
             "几何层": ("GB / km 级", "月 / 季度")}
    for y0, name, sub, c, t, bullets in bands:
        y1 = y0 + 210
        L.box(56, y0, 1520, y1, 18, WHITE, LINE, 1.6)
        L.box(76, y0 + 22, 396, y0 + 168, 12, (250, 251, 253), LINE, 1.2)
        L.text(420, y0 + 26, name, 22, c, "la", True)
        L.text(420, y0 + 60, sub, 15, MUTED, maxw=1060)
        for j, b in enumerate(bullets):
            by = y0 + 104 + j * 36
            L.d.ellipse([(432) * S, (by + 8) * S, (438) * S, (by + 14) * S], fill=c)
            L.text(450, by, b, 15, SLATE, maxw=680)

        st = stats[name]
        L.line([(1200, y0 + 26), (1200, y1 - 26)], HAIR, 1.4)
        L.text(1500, y0 + 58, "体量", 12, FAINT, "ra")
        L.text(1500, y0 + 82, st[0], 15.5, c, "ra", True)
        L.text(1500, y0 + 126, "更新频率", 12, FAINT, "ra")
        L.text(1500, y0 + 150, st[1], 15.5, c, "ra", True)

        cx0, cy0 = 76, y0 + 22
        if name == "拓扑层":
            pts = [(cx0 + 70, cy0 + 74), (cx0 + 150, cy0 + 52), (cx0 + 150, cy0 + 108),
                   (cx0 + 240, cy0 + 74), (cx0 + 240, cy0 + 122)]
            for a, b in [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]:
                L.line([pts[a], pts[b]], (203, 213, 225), 2.4)
            for x, y in pts:
                L.d.ellipse([(x - 9) * S, (y - 9) * S, (x + 9) * S, (y + 9) * S], fill=WHITE, outline=TEAL, width=int(2.2 * S))
        elif name == "语义层":
            L.box(cx0 + 40, cy0 + 34, cx0 + 110, cy0 + 86, 8, WHITE, (220, 38, 38), 2.4)
            L.text(cx0 + 75, cy0 + 60, "60", 20, (220, 38, 38), "mm", True)
            L.box(cx0 + 140, cy0 + 44, cx0 + 186, cy0 + 86, 8, WHITE, BLUE, 2.2)
            L.text(cx0 + 163, cy0 + 65, "P", 17, BLUE, "mm", True)
            L.dash(cx0 + 30, cy0 + 120, cx0 + 292, cy0 + 120, (203, 213, 225), 2, 9, 6)
            L.arrow(cx0 + 292, cy0 + 120, cx0 + 316, cy0 + 120, (203, 213, 225), 2, 9)
        else:
            for i in range(46):
                x = cx0 + 24 + ((i * 97) % 272)
                y = cy0 + 24 + ((i * 53) % 104)
                L.d.ellipse([(x - 2.2) * S, (y - 2.2) * S, (x + 2.2) * S, (y + 2.2) * S], fill=(148, 163, 184))
            L.line([(cx0 + 24, cy0 + 118), (cx0 + 160, cy0 + 108), (cx0 + 316, cy0 + 96)], BLUE, 2.6)
            L.line([(cx0 + 24, cy0 + 146), (cx0 + 170, cy0 + 140), (cx0 + 316, cy0 + 132)], BLUE, 2.6)

    # ---------------- 右侧漏斗 ----------------
    stages = [(206, 326, 760, "① 拓扑先筛", "候选车道 200 → 5 条", TEAL, TEAL_T),
              (352, 472, 600, "② 语义对齐", "路牌、箭头当锚点，先粗对齐", ORANGE, ORANGE_T),
              (498, 618, 440, "③ 几何精配", "点云 ICP，压到 3–5 cm", BLUE, BLUE_T)]
    for y0, y1, w, t1, t2, c, t in stages:
        x0 = 1950 - w / 2
        L.box(x0, y0, x0 + w, y1, 16, t, c, 1.6)
        L.text(1950, y0 + 30, t1, 19, c, "mm", True)
        L.text(1950, y0 + 64, t2, 14.5, SLATE, "mm", maxw=w - 40)
    for y in (330, 476):
        L.arrow(1950, y, 1950, y + 16, (148, 163, 184), 2, 9)

    cw = L.tw("位姿 (x, y, θ) → 进融合", 16) + 44
    L.box(1950 - cw / 2, 640, 1950 + cw / 2, 684, 12, INK, INK, 0)
    L.text(1950, 662, "位姿 (x, y, θ) → 进融合", 16, WHITE, "mm", True)

    L.text(1590, 720, "本质：用三层信息，把 6 自由度的搜索一层层缩小。", 15, MUTED)
    L.text(1590, 750, "只靠几何，等于在 200 条车道里盲配，慢且容易配错。", 15, MUTED)

    # ---------------- 三条结论 ----------------
    notes = [("只用几何", ["慢；长直通道、重复纹理处容易误配。"]),
             ("只用拓扑 + 语义", ["快；但精度到不了车道级，只能给粗位姿。"]),
             ("三层递进", ["先筛后配，又快又稳 —— 这才是量产做法。"])]
    for i, (title, body) in enumerate(notes):
        x = 60 + i * 740
        L.box(x, 892, x + 740, 1012, 14, WHITE, LINE, 1.6)
        L.text(x + 24, 916, title, 18, INK, "la", True)
        for j, b in enumerate(body):
            L.text(x + 24, 956 + j * 32, b, 14.5, SLATE, maxw=696)

    L.text(56, H - 40, "图 7 · 高精地图三层：几何 / 语义 / 拓扑 与匹配顺序", 18, FAINT)
    L.text(2344, H - 40, "nxcstudy_ai · 三维视觉 · 2026-11", 18, FAINT, "ra")
    L.save(path)


# ======================================================================
# 图 8  置信度 → 下游
# ======================================================================
def fig8(path):
    W, H = 2400, 1180
    L = Lay(W, H)
    L.text(56, 40, "定位的置信度，怎么变成规划的动作", 44, INK, "la", True)
    L.text(56, 104, "协方差不是给算法看的数字：它要变成车速、车距和接管请求", 21, MUTED)

    steps = [("定位输出", "x, y, θ, Σ", BLUE), ("完好性监测", "一致性 / RAIM", TEAL),
             ("降级状态机", "L0 – L3", (217, 119, 6)), ("规划约束", "偏移 / 距离 / 限速", ORANGE),
             ("执行 / 接管", "MRM 靠边停车", (220, 38, 38))]
    for i, (t1, t2, c) in enumerate(steps):
        x = 56 + i * 460
        L.box(x, 176, x + 440, 306, 14, WHITE, c, 1.8)
        L.text(x + 220, 226, t1, 19, INK, "mm", True)
        L.text(x + 220, 262, t2, 13.5, MUTED, "mm")
        if i:
            L.arrow(x - 16, 241, x - 4, 241, (148, 163, 184), 2.2, 10)

    levels = [(("L0", "正常", "σ < 0.1 m，多源交叉校验一致", "全速行驶，轨迹按正常间距规划"), GREEN),
              (("L1", "降级", "σ 0.1 – 0.5 m，或某个源权重下调", "车道内收紧偏移，跟车距离 ×1.5"), (217, 119, 6)),
              (("L2", "受限", "σ 0.5 – 2 m，或一致性检查告警", "限速 40，禁止变道，靠右行驶"), ORANGE),
              (("L3", "失效", "σ > 2 m，或校验失败 / 位置跳变", "MRM：靠边停车，请求接管"), (220, 38, 38))]

    for i, ((code, name, crit, act), c) in enumerate(levels):
        y0 = 340 + i * 170
        L.box(56, y0, 1320, y0 + 150, 14, WHITE, LINE, 1.6)
        L.line([(x0 := 60, y0 + 22), (x0, y0 + 128)], c, 5)
        L.text(84, y0 + 34, code, 28, c, "la", True)
        L.text(150, y0 + 34, name, 20, INK, "la", True)
        L.text(150, y0 + 74, crit, 14, MUTED, maxw=470)
        L.dash(590, y0 + 20, 590, y0 + 130, (226, 232, 240), 1.2, 5, 5)
        L.text(612, y0 + 26, "动作", 12, FAINT, "la", True)
        L.text(612, y0 + 56, act, 15.5, SLATE, maxw=660)

    # ---------------- 右侧阶梯图 ----------------
    cx0, cx1 = 1470, 2320
    cy0, cy1 = 372, 940

    def sx(v):
        return cx0 + (math.log10(v) + 2) / 3.0 * (cx1 - cx0)

    def sy(v):
        return cy1 - v / 140.0 * (cy1 - cy0)

    zones = [(0.01, 0.1, (5, 150, 105, 26)), (0.1, 0.5, (217, 119, 6, 26)),
             (0.5, 2, (234, 88, 12, 26)), (2, 10, (220, 38, 38, 26))]
    for a, b, col in zones:
        ov = Image.new("RGBA", L.im.size, (0, 0, 0, 0))
        ImageDraw.Draw(ov).rectangle([sx(a) * S, cy0 * S, sx(b) * S, cy1 * S], fill=col)
        L.im.alpha_composite(ov)

    L.text(1470, 344, "允许车速（示意）", 14, MUTED, "la", True)
    L.line([(cx0 - 50, cy0), (cx0 - 50, cy1)], LINE, 1.4)
    L.line([(cx0 - 50, cy1), (cx1, cy1)], LINE, 1.4)
    for v in (0, 40, 80, 130):
        L.line([(cx0 - 50, sy(v)), (cx1, sy(v))], (238, 242, 247), 1.2)
        L.text(cx0 - 62, sy(v), str(v), 13, FAINT, "rm")
    L.text(cx0 - 62, cy0 - 22, "km/h", 12, FAINT, "rm")
    for v in (0.01, 0.1, 0.5, 1, 2, 10):
        L.line([(sx(v), cy0), (sx(v), cy1)], (238, 242, 247), 1.2)
        L.text(sx(v), cy1 + 8, ("%.2f" % v).rstrip("0").rstrip("."), 12.5, FAINT, "ma")
    L.text(cx1, cy1 + 34, "定位标准差 σ（m，对数轴）", 13.5, MUTED, "ra")

    curve = [(0.01, 130), (0.1, 130), (0.1, 80), (0.5, 80), (0.5, 40), (2, 40), (2, 0), (10, 0)]
    ov = Image.new("RGBA", L.im.size, (0, 0, 0, 0))
    ImageDraw.Draw(ov).polygon([(sx(a) * S, sy(b) * S) for a, b in curve] +
                               [(cx1 * S, cy1 * S), (sx(0.01) * S, cy1 * S)], fill=(15, 23, 42, 16))
    L.im.alpha_composite(ov)
    L.line([(sx(a), sy(b)) for a, b in curve], INK, 3)
    for i, (txt, v) in enumerate([("130", 0.05), ("80", 0.2), ("40", 1.0), ("0", 4.0)]):
        L.text(sx(v), sy([130, 80, 40, 0][i]) - 18, txt, 12.5, INK, "ma", True)

    # ---------------- 一句话 ----------------
    L.box(56, 1036, 2344, 1114, 16, (241, 245, 249), LINE, 1.4)
    L.text(88, 1062, "定位模块对外的产出不是 (x, y)，而是 (x, y, Σ)：带了方差的位姿，下游才敢用它降级。",
           19, INK, "la", True, maxw=2220)

    L.text(56, H - 38, "图 8 · 从定位置信度到规划动作：四级降级", 18, FAINT)
    L.text(2344, H - 38, "nxcstudy_ai · 三维视觉 · 2026-11", 18, FAINT, "ra")
    L.save(path)


# ======================================================================
# 图 9  地库 AVP 的 UWB 布点
# ======================================================================
def fig9(path):
    W, H = 2400, 1160
    L = Lay(W, H)
    L.text(56, 40, "地库 AVP：UWB 锚点怎么布，才敢把车交出去", 44, INK, "la", True)
    L.text(56, 104, "同一个车位，锚点几何（GDOP）、柱子遮挡和点云退化，决定了哪一段该靠谁", 21, MUTED)

    # 车库轮廓
    L.box(56, 186, 1560, 960, 18, (252, 253, 255), (203, 213, 225), 2)

    # 车位
    for i in range(5):
        y = 232 + i * 148
        L.box(76, y, 176, y + 118, 8, (246, 248, 252), (226, 232, 240), 1.2)
        L.box(1440, y, 1540, y + 118, 8, (246, 248, 252), (226, 232, 240), 1.2)
    L.box(1440, 232, 1540, 350, 8, TEAL_T, TEAL, 1.8)
    L.text(1490, 291, "目标", 14, TEAL, "mm", True)

    # 点云退化区
    L.box(220, 560, 760, 800, 12, (255, 247, 237), (234, 88, 12), 1.6)
    L.text(240, 578, "点云退化区：长直、侧向无特征", 13.5, (154, 52, 18), "la", True)

    # 柱子
    for px, py in [(360, 400), (360, 700), (1080, 400), (1080, 700), (680, 400)]:
        L.box(px, py, px + 34, py + 34, 4, (226, 232, 240), (148, 163, 184), 1.4)

    # 行驶路径
    route = [(480, 958), (480, 620), (1240, 620), (1240, 290), (1396, 290)]
    for i in range(len(route) - 1):
        L.dash(route[i][0], route[i][1], route[i + 1][0], route[i + 1][1], (30, 41, 59), 3.2, 12, 8)
    L.arrow(1360, 290, 1396, 290, (30, 41, 59), 3.2, 12)
    L.d.ellipse([(472) * S, (950) * S, (488) * S, (966) * S], fill=(30, 41, 59))

    # UWB 锚点
    for ax, ay in [(200, 222), (1420, 222), (200, 928), (1420, 928), (810, 222), (810, 928)]:
        L.d.polygon([(ax * S, (ay - 13) * S), ((ax + 13) * S, ay * S), (ax * S, (ay + 13) * S),
                     ((ax - 13) * S, ay * S)], fill=BLUE, outline=WHITE, width=int(2 * S))

    # 分区线
    for x in (600, 1030):
        L.dash(x, 200, x, 946, (203, 213, 225), 1.4, 7, 7)
    for cx_, txt in [(388, "A 区 · 可见锚点 4"), (815, "B 区 · 可见锚点 3"), (1235, "C 区 · 可见锚点 4")]:
        L.text(cx_, 250, txt, 13, MUTED, "ma", True)

    # 阶段标记
    for n, (mx, my) in enumerate([(480, 930), (480, 640), (1240, 380)], start=1):
        L.d.ellipse([(mx - 15) * S, (my - 15) * S, (mx + 15) * S, (my + 15) * S], fill=WHITE, outline=INK, width=int(2 * S))
        L.text(mx, my, str(n), 15, INK, "mm", True)

    # 图例（图形都手绘，避免字体缺字）
    L.d.polygon([(880 * S, 866 * S), (890 * S, 876 * S), (880 * S, 886 * S), (870 * S, 876 * S)],
                fill=BLUE, outline=WHITE, width=int(2 * S))
    L.text(900, 868, "UWB 锚点", 13, MUTED)
    L.dash(1000, 876, 1044, 876, INK, 3.0, 8, 6)
    L.text(1054, 868, "行驶路径", 13, MUTED)
    L.box(1150, 864, 1182, 890, 4, (255, 247, 237), (234, 88, 12), 1.4)
    L.text(1192, 868, "点云退化区", 13, MUTED)

    # 三阶段说明
    ph = [("① 入口", "GNSS 尚存，绝对定位仍在", BLUE),
          ("② 长直通道", "点云退化 → 切 UWB 主导", (234, 88, 12)),
          ("③ 车位", "点云重定位 + UWB 复核", TEAL)]
    x = 60
    for i, (t1, t2, c) in enumerate(ph):
        L.box(x, 986, x + 470, 1052, 12, WHITE, c, 1.4)
        L.text(x + 18, 998, t1, 15, c, "la", True)
        L.text(x + 18, 1022, t2, 13, SLATE, maxw=440)
        if i < 2:
            L.arrow(x + 474, 1019, x + 492, 1019, (148, 163, 184), 2, 9)
        x += 500

    # 右侧三条
    cards = [("① 锚点几何（GDOP）",
              ["锚点共线、或全挤在车头一侧时，横向解算的方差会被放大。", "四个角各一个，是底线配置；只有两个角，就别谈厘米级。",
               "经验值：锚点间距 ≈ 分区对角线的 0.6–0.8 倍。"]),
             ("② 遮挡盲区",
              ["柱子和车身会挡住直射路径：每个分区至少保证 3 个可见锚点，", "否则那一格只能靠点云；锚点挂高 2.5 m 以上，遮挡会小很多。",
               "地面反射和多径会让「看得见」也不准，布点时留裕量。"]),
             ("③ 切换区",
              ["UWB 与点云配准要有一段重叠区，在重叠区里做一致性检查：", "两套位姿差 < 10 cm 且持续 1 s，才允许切换。",
               "切换失败要能退回上一个状态，别在原地反复横跳。"])]
    for i, (title, body) in enumerate(cards):
        y0 = 200 + i * 250
        L.box(1620, y0, 2340, y0 + 200, 14, WHITE, LINE, 1.6)
        L.text(1644, y0 + 20, title, 18, INK, "la", True)
        for j, b in enumerate(body):
            L.text(1644, y0 + 62 + j * 30, b, 14, SLATE, maxw=672)

    L.text(56, H - 38, "图 9 · 地库 AVP：UWB 锚点布设、点云退化与切换区", 18, FAINT)
    L.text(2344, H - 38, "nxcstudy_ai · 三维视觉 · 2026-11", 18, FAINT, "ra")
    L.save(path)
if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    os.makedirs(out, exist_ok=True)
    fig1(os.path.join(out, "2026-11-自动驾驶定位-图1-技术全景.png"))
    fig2(os.path.join(out, "2026-11-自动驾驶定位-图2-分层架构.png"))
    fig3(os.path.join(out, "2026-11-自动驾驶定位-图3-隧道时序.png"))
    fig4(os.path.join(out, "2026-11-自动驾驶定位-图4-坐标系变换链.png"))
    fig5(os.path.join(out, "2026-11-自动驾驶定位-图5-递推与因子图.png"))
    fig6(os.path.join(out, "2026-11-自动驾驶定位-图6-松紧耦合.png"))
    fig7(os.path.join(out, "2026-11-自动驾驶定位-图7-地图三层.png"))
    fig8(os.path.join(out, "2026-11-自动驾驶定位-图8-置信度降级.png"))
    fig9(os.path.join(out, "2026-11-自动驾驶定位-图9-地库AVP.png"))
