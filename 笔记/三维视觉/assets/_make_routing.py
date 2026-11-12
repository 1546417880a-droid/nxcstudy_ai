import math
from PIL import Image, ImageDraw, ImageFont

W, H = 2400, 1560
S = 2
FONT = "/System/Library/Fonts/Hiragino Sans GB.ttc"

INK = (15, 23, 42)
MUTED = (100, 116, 139)
FAINT = (148, 163, 184)
LINE = (226, 232, 240)
BG = (248, 250, 252)
WHITE = (255, 255, 255)

BLUE = (37, 99, 235)
BLUE_BG = (239, 246, 255)
GREEN = (5, 150, 105)
GREEN_BG = (236, 253, 245)
ORANGE = (234, 88, 12)
ORANGE_BG = (255, 247, 237)

_fc = {}
def font(sz, bold=False):
    k = (sz, bold)
    if k not in _fc:
        _fc[k] = ImageFont.truetype(FONT, int(sz * S), index=2 if bold else 0)
    return _fc[k]

class Lay:
    def __init__(self, im):
        self.im = im
        self.d = ImageDraw.Draw(im)
    def text(self, x, y, s, sz=24, color=INK, anchor="la", bold=False):
        self.d.text((x * S, y * S), s, font=font(sz, bold), fill=color, anchor=anchor)
    def line(self, pts, color, width=2):
        self.d.line([(x * S, y * S) for x, y in pts], fill=color, width=int(width * S), joint="curve")
    def box(self, x0, y0, x1, y1, r=14, fill=WHITE, outline=LINE, width=2):
        self.d.rounded_rectangle([x0 * S, y0 * S, x1 * S, y1 * S], radius=r * S,
                                 fill=fill, outline=outline, width=int(width * S))
    def dash(self, pts, color, width=2, on=10, off=7):
        for i in range(len(pts) - 1):
            (x0, y0), (x1, y1) = pts[i], pts[i + 1]
            Ln = math.hypot(x1 - x0, y1 - y0)
            n = max(1, int(Ln / (on + off)))
            for k in range(n + 1):
                t0 = min(1, k * (on + off) / Ln)
                t1 = min(1, (k * (on + off) + on) / Ln)
                if t1 <= t0:
                    continue
                self.line([(x0 + (x1 - x0) * t0, y0 + (y1 - y0) * t0),
                           (x0 + (x1 - x0) * t1, y0 + (y1 - y0) * t1)], color, width)

def bez(p0, p1, p2, p3, n=48):
    out = []
    for i in range(n + 1):
        t = i / n
        u = 1 - t
        out.append((u**3*p0[0] + 3*u*u*t*p1[0] + 3*u*t*t*p2[0] + t**3*p3[0],
                    u**3*p0[1] + 3*u*u*t*p1[1] + 3*u*t*t*p2[1] + t**3*p3[1]))
    return out

img = Image.new("RGBA", (W * S, H * S), BG + (255,))
L = Lay(img)

L.text(60, 34, "传感器不是都为感知服务 —— 三条 KPI，三个去处", 46, INK, "la", True)
L.text(60, 98, "每颗传感器都有「主职」和「兼职」；实线 = 主职，虚线 = 兼职。感知只是外感受的主场",
       24, MUTED)

# ---------- geometry band ----------
L.box(60, 140, 2340, 212, r=14, fill=(241, 245, 249), outline=(203, 213, 225), width=2)
L.text(84, 156, "② 传感器数据处理", 22, (51, 65, 85), "la", True)
L.text(300, 160, "对齐：时间同步 · 去畸变 · 外参统一（每颗都过）   增广：双目→点云 · 单目→深度 · 预积分（按需）",
       16, MUTED)

# ---------- sensors ----------
CARD_H, STEP = 70, 80
SX0, SX1 = 90, 730

def group(top, cards, color, tint, title, sub):
    h = 64 + len(cards) * STEP - (STEP - CARD_H) + 10
    L.box(60, top, 760, top + h, r=18, fill=tint, outline=None, width=0)
    L.text(84, top + 12, title, 21, color, "la", True)
    L.text(84, top + 38, sub, 15, MUTED)
    ys = []
    for i, (t, s) in enumerate(cards):
        y0 = top + 64 + i * STEP
        L.box(SX0, y0, SX1, y0 + CARD_H, r=12, fill=WHITE, outline=(226, 232, 240), width=2)
        L.d.rounded_rectangle([SX0 * S, (y0 + 8) * S, (SX0 + 6) * S, (y0 + CARD_H - 8) * S],
                              radius=3 * S, fill=color)
        L.text(112, y0 + 9, t, 22, INK, "la", True)
        L.text(112, y0 + 40, s, 14, MUTED)
        ys.append(y0 + CARD_H // 2)
    return ys

y_cam, y_lidar, y_radar, y_sonar = group(230, [
    ("相机", "单目 · 双目 · RGB-D · 10~60 Hz"),
    ("LiDAR", "机械 / 固态 · 直接测距 · 10~20 Hz"),
    ("毫米波雷达", "远距 · 测速 · 雨雾鲁棒 · 10~20 Hz"),
    ("超声波", "近距 0.2~5 m · 泊车"),
], BLUE, BLUE_BG, "外感受", "问：外面有什么？ → 主职去感知")

y_imu, y_wheel, y_gnss = group(634, [
    ("IMU", "加速度 · 角速度 · 100~1000 Hz"),
    ("轮速计", "4 轮 · 尺度 · 里程计"),
    ("GNSS / RTK", "绝对位置 · 1~10 Hz"),
], GREEN, GREEN_BG, "本体感受", "问：我在哪、怎么动？ → 主职去定位")

y_can, = group(958, [
    ("底盘 CAN", "车速 · 转角 · 制动 · 档位 · 故障码（命令下行 ←→ 状态上行）"),
], ORANGE, ORANGE_BG, "底盘 CAN", "问：车听没听话？ → 主职去控制闭环")

# ---------- destinations ----------
DX0, DX1 = 1560, 2340
def dest(y0, h, color, title, lines):
    L.box(DX0, y0, DX1, y0 + h, r=18, fill=WHITE, outline=color, width=2)
    L.d.rounded_rectangle([DX0 * S, y0 * S, (DX0 + 9) * S, (y0 + h) * S], radius=4 * S, fill=color)
    L.text(DX0 + 34, y0 + 18, title, 30, INK, "la", True)
    yy = y0 + 68
    for i, s in enumerate(lines):
        L.text(DX0 + 34, yy, s, 20 if i else 20, MUTED if i else INK)
        yy += 32

dest(300, 210, BLUE, "③ 感知", [
    "输出：目标 · 障碍 · 占用 · 可通行",
    "主力：相机 · LiDAR · 毫米波 · 超声波",
    "兼职：IMU（去畸变 / 运动补偿）",
])
dest(580, 210, GREEN, "定位与建图", [
    "输出：pose + 协方差 + 地图",
    "主力：LiDAR · 相机 · IMU · 轮速 · GNSS",
    "兼职：CAN（转角 → 运动模型）",
])
dest(860, 220, ORANGE, "④ 决策 / ⑤ 控制", [
    "输出：轨迹跟踪 · 闭环反馈 · 诊断",
    "主力：CAN（车速/转角/制动/档位）",
    "兼职：IMU（姿态）· 轮速（速度前馈）",
])

# ---------- edges ----------
def edge(sy, dy, color, primary=True, label=None):
    p = bez((740, sy), (1040, sy), (1260, dy), (1552, dy))
    if primary:
        L.line(p, color, 3)
    else:
        L.dash(p, color, 2, on=9, off=8)
    if label:
        mx, my = p[len(p) // 2]
        w = L.d.textlength(label, font=font(13)) / S
        L.d.rounded_rectangle([(mx - w / 2 - 7) * S, (my - 11) * S,
                               (mx + w / 2 + 7) * S, (my + 11) * S],
                              radius=6 * S, fill=WHITE, outline=(226, 232, 240), width=1 * S)
        L.text(mx, my, label, 13, MUTED, "mm")

edge(y_cam, 405, BLUE)
edge(y_cam, 685, BLUE)
edge(y_lidar, 405, BLUE)
edge(y_lidar, 685, BLUE)
edge(y_radar, 405, BLUE)
edge(y_sonar, 405, BLUE)
edge(y_imu, 685, GREEN)
edge(y_wheel, 685, GREEN)
edge(y_wheel, 970, GREEN)
edge(y_gnss, 685, GREEN)
edge(y_can, 970, ORANGE)

edge(y_imu, 405, GREEN, False, "去畸变")
edge(y_can, 685, ORANGE, False, "转角约束")

# ---------- bottom ----------
L.box(60, 1150, 2340, 1520, r=18, fill=WHITE, outline=LINE, width=2)
L.text(90, 1162, "两条容易搞反的结论", 22, INK, "la", True)
L.text(90, 1202, "结论 1：感知只是「外感受」的 KPI，不是所有传感器的目的地 —— 本体的 KPI 是定位，CAN 的 KPI 是控制闭环。",
       19, INK)
L.text(90, 1240, "结论 2：同一颗传感器常被多处消费 —— IMU 进标定/感知/定位/控制，轮速进定位/控制，相机也参与定位与重定位。",
       19, INK)
L.text(90, 1278, "所以 ① 物理层是「全栈的数据源头」，不是「感知的输入口」—— 这是整张图最容易搞反的一处。",
       19, (154, 52, 18))

chips = [
    ("外感受", "感知", BLUE, 90),
    ("本体感受", "定位", GREEN, 620),
    ("底盘 CAN", "控制闭环", ORANGE, 1150),
]
for name, kpi, col, x in chips:
    L.box(x, 1326, x + 470, 1428, r=14, fill=WHITE, outline=col, width=2)
    L.text(x + 24, 1342, name, 18, MUTED)
    L.text(x + 24, 1372, "KPI = " + kpi, 26, col, "la", True)
L.text(1700, 1342, "一句话记住", 17, MUTED, "la", True)
L.text(1700, 1372, "外感受向外看，本体向内看，", 20, INK)
L.text(1700, 1404, "CAN 向下看 —— 三条 KPI 不重叠。", 20, INK)

L.text(1200, 1470, "记忆：眼睛负责「有什么」，前庭负责「我在哪」，脊髓负责「做没做到」", 18, MUTED, "ma")

out = img.convert("RGB").resize((W, H), Image.LANCZOS)
out.save("/Users/ningxincen/Downloads/code/nxcstudy_ai/assets/传感器去向.png")
print("saved", out.size, "centers:", y_cam, y_lidar, y_radar, y_sonar, y_imu, y_wheel, y_gnss, y_can)
