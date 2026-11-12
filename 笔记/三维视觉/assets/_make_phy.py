import math
from PIL import Image, ImageDraw, ImageFont, ImageChops

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
BLUE_D = (29, 78, 216)
BLUE_L = (219, 234, 254)
SKY_D = (3, 105, 161)
GREEN = (5, 150, 105)
GREEN_L = (209, 250, 229)
TEAL = (13, 148, 136)
ORANGE = (234, 88, 12)
ORANGE_D = (154, 52, 18)

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
    def box(self, x0, y0, x1, y1, r=16, fill=WHITE, outline=LINE, width=2):
        self.d.rounded_rectangle([x0 * S, y0 * S, x1 * S, y1 * S], radius=r * S,
                                 fill=fill, outline=outline, width=int(width * S))
    def pie(self, px, py, r, a0, a1, color, alpha):
        self.d.pieslice([(px - r) * S, (py - r) * S, (px + r) * S, (py + r) * S],
                        a0, a1, fill=color + (alpha,))
    def arrow(self, x0, y0, x1, y1, color, width=3, head=14):
        self.d.line([(x0 * S, y0 * S), (x1 * S, y1 * S)], fill=color, width=int(width * S))
        ang = math.atan2(y1 - y0, x1 - x0)
        L, Wd = head, head * 0.45
        p2 = (x1 - L * math.cos(ang) + Wd * math.sin(ang), y1 - L * math.sin(ang) - Wd * math.cos(ang))
        p3 = (x1 - L * math.cos(ang) - Wd * math.sin(ang), y1 - L * math.sin(ang) + Wd * math.cos(ang))
        self.d.polygon([(x1 * S, y1 * S), (p2[0] * S, p2[1] * S), (p3[0] * S, p3[1] * S)], fill=color)
    def dot(self, x, y, color, r=4):
        self.d.ellipse([(x - r) * S, (y - r) * S, (x + r) * S, (y + r) * S], fill=color)

img = Image.new("RGBA", (W * S, H * S), BG + (255,))
main = Lay(img)

# ============ header ============
main.text(60, 34, "物理层为什么分三类：向外看 · 向内看 · 向下看", 46, INK, "la", True)
main.text(60, 98, "不是分类学，而是三个不同的问题 —— 测的对象不同、参考系不同、失效模式不同、下游也不同",
          24, MUTED)

# ============ scene layer ============
scene = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
sc = Lay(scene)

PX0, PY0, PX1, PY1 = 60, 150, 1520, 1140
sc.box(PX0, PY0, PX1, PY1, r=20, fill=WHITE, outline=LINE, width=2)
sc.text(84, 166, "俯视图 · 示意：浅蓝 = 相机 FOV ｜ 青 = 毫米波 ｜ 淡蓝扇形 = 超声波 ｜ 虚线环 = LiDAR 360°（半径已压缩）", 16, FAINT)

cx, cy = 800, 655

# beams on an overlay so they blend with the panel
beams = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
bl = Lay(beams)
bl.pie(cx + 245, cy, 430, -32, 32, (59, 130, 246), 40)
bl.pie(cx - 245, cy, 400, 148, 212, (59, 130, 246), 40)
bl.pie(cx, cy - 105, 300, 228, 312, (59, 130, 246), 40)
bl.pie(cx, cy + 105, 300, 48, 132, (59, 130, 246), 40)
bl.pie(cx + 245, cy, 450, -8, 8, (14, 165, 233), 38)
bl.pie(cx - 245, cy, 330, 172, 188, (14, 165, 233), 38)
for (ux, uy, a0, a1) in [(cx + 250, cy - 78, -50, 0), (cx + 250, cy + 78, 0, 50),
                         (cx - 250, cy - 78, 180, 230), (cx - 250, cy + 78, 130, 180)]:
    bl.pie(ux, uy, 88, a0, a1, (96, 165, 250), 90)
scene = Image.alpha_composite(scene, beams)
sc = Lay(scene)

# grid
for gx in range(140, 1500, 80):
    sc.line([(gx, 200), (gx, 1120)], (243, 246, 251), 1)
for gy in range(200, 1120, 80):
    sc.line([(140, gy), (1500, gy)], (243, 246, 251), 1)

# cone outlines + lidar ring
for (px, py, r, a0, a1) in [(cx + 245, cy, 430, -32, 32), (cx - 245, cy, 400, 148, 212),
                            (cx, cy - 105, 300, 228, 312), (cx, cy + 105, 300, 48, 132)]:
    sc.d.arc([(px - r) * S, (py - r) * S, (px + r) * S, (py + r) * S], a0, a1,
             fill=(147, 197, 253), width=int(1.5 * S))
a = 0
while a < 360:
    sc.d.arc([(cx - 420) * S, (cy - 420) * S, (cx + 420) * S, (cy + 420) * S],
             start=a, end=a + 4, fill=(96, 165, 250), width=int(2 * S))
    a += 7

# ============ car ============
d = sc.d
d.rounded_rectangle([(cx - 250) * S, (cy - 100) * S, (cx + 250) * S, (cy + 100) * S],
                    radius=50 * S, fill=(241, 245, 249), outline=(148, 163, 184), width=int(3 * S))
d.polygon([((cx + 55) * S, (cy - 86) * S), ((cx + 140) * S, (cy - 66) * S),
           ((cx + 140) * S, (cy + 66) * S), ((cx + 55) * S, (cy + 86) * S)],
          fill=(226, 236, 248))
sc.line([(cx - 60, cy - 92), (cx - 60, cy + 92)], (203, 213, 225), 2)

for (wx, wy) in [(cx + 140, cy - 100), (cx + 140, cy + 100), (cx - 150, cy - 100), (cx - 150, cy + 100)]:
    d.rounded_rectangle([(wx - 38) * S, (wy - 13) * S, (wx + 38) * S, (wy + 13) * S],
                        radius=8 * S, fill=(71, 85, 105))
    d.ellipse([(wx - 9) * S, (wy - 9) * S, (wx + 9) * S, (wy + 9) * S], fill=GREEN)
    d.ellipse([(wx - 4) * S, (wy - 4) * S, (wx + 4) * S, (wy + 4) * S], fill=GREEN_L)

d.ellipse([(cx - 20) * S, (cy - 30) * S, (cx + 40) * S, (cy + 30) * S],
          fill=BLUE_L, outline=BLUE, width=int(3 * S))
d.ellipse([(cx + 1) * S, (cy - 9) * S, (cx + 19) * S, (cy + 9) * S], fill=BLUE)

d.rounded_rectangle([(cx - 105) * S, (cy - 22) * S, (cx - 60) * S, (cy + 22) * S],
                    radius=8 * S, fill=GREEN_L, outline=GREEN, width=int(2.5 * S))
sc.text(cx - 82, cy, "IMU", 13, GREEN, "mm", True)

d.rounded_rectangle([(cx - 215) * S, (cy - 12) * S, (cx - 175) * S, (cy + 12) * S],
                    radius=6 * S, fill=(204, 251, 241), outline=TEAL, width=int(2 * S))
sc.text(cx - 195, cy + 26, "GNSS 天线", 12, TEAL, "mm")

# ============ satellite ============
sat_x, sat_y = 340, 300
ex, ey = cx - 195, cy - 14
dx, dy = sat_x + 34, sat_y + 16
dist = math.hypot(dx - ex, dy - ey)
n = int(dist / 13)
for i in range(0, n, 2):
    t0, t1 = i / n, min((i + 2) / n, 1)
    sc.line([(ex + (dx - ex) * t0, ey + (dy - ey) * t0),
             (ex + (dx - ex) * t1, ey + (dy - ey) * t1)], TEAL, 1.5)
d.rounded_rectangle([(sat_x - 8) * S, (sat_y - 14) * S, (sat_x + 22) * S, (sat_y + 14) * S],
                    radius=4 * S, fill=(240, 253, 250), outline=TEAL, width=int(2 * S))
d.rectangle([(sat_x - 56) * S, (sat_y - 7) * S, (sat_x - 12) * S, (sat_y + 7) * S],
            fill=(153, 246, 228), outline=TEAL, width=int(1.5 * S))
d.rectangle([(sat_x + 26) * S, (sat_y - 7) * S, (sat_x + 70) * S, (sat_y + 7) * S],
            fill=(153, 246, 228), outline=TEAL, width=int(1.5 * S))
sc.text(sat_x + 78, sat_y - 26, "GNSS / RTK", 20, TEAL, "la", True)
sc.text(sat_x + 78, sat_y - 2, "与卫星测距 → 绝对位置（1~10 Hz）", 14, MUTED)

# ============ labels ============
def label(x, y, title, sub=None, color=BLUE_D, anchor="la", sz=20):
    sc.text(x, y, title, sz, color, anchor, True)
    if sub:
        sc.text(x, y + 22, sub, 13, MUTED, anchor)

label(cx + 330, cy - 10, "相机 前", "60° FOV · 10~60 Hz", BLUE_D)
label(cx - 330, cy + 40, "相机 后", None, BLUE_D, "ra")
label(cx + 40, cy - 300, "相机 左", None, BLUE_D)
label(cx + 40, cy + 272, "相机 右", None, BLUE_D)

sc.line([(cx - 170, cy - 296), (cx - 6, cy - 34)], (191, 219, 254), 1.5)
label(cx - 178, cy - 320, "LiDAR", "360° · 直接测距（米）", BLUE_D)

sc.line([(cx + 452, cy - 56), (cx + 380, cy - 34)], (186, 230, 253), 1.5)
label(cx + 458, cy - 92, "毫米波 前", "远距 · 测速", SKY_D, sz=19)
sc.line([(cx - 420, cy + 26), (cx - 300, cy + 10)], (186, 230, 253), 1.5)
sc.text(cx - 426, cy + 8, "毫米波 后", 19, SKY_D, "ra", True)

sc.line([(cx + 262, cy + 100), (cx + 214, cy + 86)], (191, 219, 254), 1.5)
label(cx + 268, cy + 88, "超声波", "近距 · 泊车", BLUE, sz=18)

sc.line([(cx - 150, cy - 140), (cx - 96, cy - 26)], (167, 243, 208), 1.5)
label(cx - 158, cy - 166, "IMU", "加速度 · 角速度（200~1000 Hz）", GREEN)

sc.line([(cx - 240, cy + 166), (cx - 158, cy + 112)], (167, 243, 208), 1.5)
label(cx - 248, cy + 160, "轮速计 ×4", "尺度 · 里程计", GREEN)

sc.line([(cx + 186, cy - 146), (cx + 152, cy - 112)], (167, 243, 208), 1.5)
sc.text(cx + 192, cy - 164, "前轮转角", 14, GREEN, "la", True)

# ============ frame note (bottom-left of scene) ============
sc.line([(110, 1046), (110, 1000)], FAINT, 2)
sc.line([(110, 1046), (156, 1046)], FAINT, 2)
sc.d.polygon([(104 * S, 994 * S), (110 * S, 1004 * S), (116 * S, 994 * S)], fill=FAINT)
sc.d.polygon([(162 * S, 1040 * S), (152 * S, 1046 * S), (162 * S, 1052 * S)], fill=FAINT)
sc.text(174, 988, "世界系 ENU / UTM：固定不动，pose 的基准", 16, MUTED)
sc.text(174, 1016, "车体系：原点在后轴中心，x 朝前、y 朝左", 16, MUTED)

# ============ legend ============
def chip(x, y, color, s):
    sc.d.rounded_rectangle([x * S, (y - 11) * S, (x + 18) * S, (y + 11) * S], radius=4 * S, fill=color)
    sc.text(x + 28, y, s, 19, INK, "lm")

chip(90, 214, BLUE, "外感受")
chip(250, 214, GREEN, "本体感受")
chip(440, 214, ORANGE, "底盘 CAN（通道，不是感受器）")

# mask scene to panel
mask = Image.new("L", (W * S, H * S), 0)
ImageDraw.Draw(mask).rounded_rectangle([PX0 * S, PY0 * S, PX1 * S, PY1 * S], radius=20 * S, fill=255)
scene.putalpha(ImageChops.multiply(scene.getchannel("A"), mask))
img = Image.alpha_composite(img, scene)
main = Lay(img)

# ============ right cards ============
CARD_X0, CARD_W = 1560, 780
def card(y0, y1, accent, zh, en, rows, out, meta):
    main.box(CARD_X0, y0, CARD_X0 + CARD_W, y1, r=18, fill=WHITE, outline=LINE, width=2)
    d2 = main.d
    d2.rounded_rectangle([CARD_X0 * S, y0 * S, (CARD_X0 + 9) * S, y1 * S], radius=4 * S, fill=accent)
    main.text(CARD_X0 + 34, y0 + 20, zh, 34, INK, "la", True)
    tw = d2.textlength(zh, font=font(34, True)) / S
    main.text(CARD_X0 + 34 + tw + 14, y0 + 34, en, 16, accent)
    yy = y0 + 92
    for k, v in rows:
        main.text(CARD_X0 + 34, yy, k, 21, MUTED)
        main.text(CARD_X0 + 96, yy, v, 22, INK)
        yy += 40
    main.text(CARD_X0 + 34, y0 + 250, out, 23, accent, "la", True)
    main.text(CARD_X0 + 34, y0 + 286, meta, 16, MUTED)

card(150, 464, BLUE, "外感受", "EXTEROCEPTIVE", [
    ("问", "外面有什么？"),
    ("测", "光 / 点 / 回波"),
    ("系", "世界 ←→ 车体"),
    ("频", "10~60 Hz"),
], "→  感知层：目标 · 障碍 · 可通行",
    "类比 眼·耳·皮肤    ｜    失效 = 瞎")

card(488, 802, GREEN, "本体感受", "PROPRIOCEPTION", [
    ("问", "我在哪、怎么动？"),
    ("测", "加速度·角速度·轮速·卫星"),
    ("系", "车体 → 世界"),
    ("频", "100~1000 Hz（GNSS 1~10）"),
], "→  定位建图：速度 · 姿态 · 位姿",
    "类比 内耳前庭·肌梭    ｜    失效 = 漂")

card(826, 1140, ORANGE, "底盘 CAN", "CHASSIS BUS", [
    ("问", "车听没听话？"),
    ("测", "期望  vs  实际"),
    ("系", "执行器 ←→ 车体"),
    ("频", "50~100 Hz"),
], "→  控制 / 诊断：车速·转角·制动",
    "类比 脊髓·运动神经    ｜    失效 = 开环")

# ============ CAN band ============
main.box(60, 1165, 2340, 1292, r=16, fill=(255, 251, 247), outline=(254, 215, 170), width=2)
main.text(84, 1176, "底盘 CAN 是一根总线、两个方向 —— 它是「通道」，不是感受器", 21, ORANGE_D, "la", True)

bx0, by0, bx1, by1 = 84, 1212, 300, 1274
main.box(bx0, by0, bx1, by1, r=10, fill=WHITE, outline=ORANGE, width=2)
main.text((bx0 + bx1) / 2, (by0 + by1) / 2, "规划 / 控制", 21, INK, "mm", True)

dx0, dx1 = 1900, 2316
main.box(dx0, by0, dx1, by1, r=10, fill=WHITE, outline=ORANGE, width=2)
main.text((dx0 + dx1) / 2, (by0 + by1) / 2, "EPS · ESC · VCU · 档位", 21, INK, "mm", True)

main.arrow(320, 1228, 1880, 1228, ORANGE, 3, 12)
main.arrow(1880, 1260, 320, 1260, (148, 163, 184), 3, 12)
main.text(1100, 1200, "命令下行：期望转角 / 期望加速度 / 制动压力", 17, (194, 65, 12), "mm")
main.text(1100, 1272, "状态上行：实际车速 / 转角 / 制动 / 故障码", 17, MUTED, "mm")

# ============ bottom strip ============
tiles = [
    ("①  向外看", "世界相对我", "→ 感知    缺了会撞", BLUE),
    ("②  向内看", "我相对世界", "→ 定位    缺了会漂", GREEN),
    ("③  向下看", "命令相对实际", "→ 控制    缺了会开环", ORANGE),
]
tx = 60
for (a, b, c, col) in tiles:
    main.box(tx, 1322, tx + 720, 1440, r=16, fill=WHITE, outline=LINE, width=2)
    main.text(tx + 26, 1334, a, 26, col, "la", True)
    main.text(tx + 26, 1374, b, 20, INK)
    main.text(tx + 26, 1402, c, 18, MUTED)
    tx += 740

main.text(1200, 1470, "三分是功能划分，不是分类学：GNSS 属外感受却归本体（它测「我在世界的位置」）；"
                       "CAN 是通道不是感受器（命令下行 + 状态上行共用一根总线）", 19, MUTED, "ma")
main.text(1200, 1508, "记忆钩子 —— 眼耳皮肤 ｜ 内耳前庭与肌梭 ｜ 脊髓与关节反馈", 22, INK, "ma", True)

out = img.convert("RGB").resize((W, H), Image.LANCZOS)
out.save("/Users/ningxincen/Downloads/code/nxcstudy_ai/assets/物理层三分.png")
print("saved", out.size)
