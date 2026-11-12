# -*- coding: utf-8 -*-
"""共享画布：PIL 直绘、2× 超采样、A4 级导出。

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
