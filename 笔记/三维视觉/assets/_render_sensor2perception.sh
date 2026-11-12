#!/bin/bash
# 重出「传感器到感知」系列：图 1 + 架构讲解 PDF + 疑问记录 PDF。
# 依赖：Google Chrome；裁边需要 Pillow（缺省时退化为不裁边）。
# 用法：./_render_sensor2perception.sh [fig|pdf|all]     默认 all
set -euo pipefail
cd "$(dirname "$0")"

CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
PYTHON="${PYTHON:-python3}"
PROFILE="${TMPDIR:-/tmp}/chrome-s2p-profile"
MODE="${1:-all}"

FIG="2026-11-传感器到感知-图1-从物理量到感知.png"
NOTE_PDF="../2026-11-传感器到感知-架构讲解.pdf"
QA_PDF="../2026-11-传感器到感知-疑问记录.pdf"

render_fig() {
  local raw="${TMPDIR:-/tmp}/s2p_fig_raw.png"
  echo "→ 截图 $FIG"
  "$CHROME" --headless --disable-gpu --hide-scrollbars --no-sandbox \
    --disable-breakpad --disable-crash-reporter --no-first-run \
    --user-data-dir="$PROFILE" --force-device-scale-factor=2 \
    --window-size=1700,1300 --screenshot="$raw" \
    "file://$PWD/_make_sensor2perception_fig.html" >/dev/null 2>&1

  "$PYTHON" - "$raw" "$FIG" <<'PY' || { echo "!! Pillow 不可用，直接拷贝未裁边版本"; cp "$raw" "$FIG"; }
import sys
from PIL import Image
raw, out = sys.argv[1], sys.argv[2]
im = Image.open(raw).convert("RGB")
bg = (232, 237, 244)                    # 页面底色 #E8EDF4
px = im.load(); W, H = im.size
def hit(c): return abs(c[0]-bg[0]) + abs(c[1]-bg[1]) + abs(c[2]-bg[2]) > 8
minx, miny, maxx, maxy = W, H, 0, 0
for y in range(0, H, 2):
    for x in range(0, W, 2):
        if hit(px[x, y]):
            minx = min(minx, x); miny = min(miny, y)
            maxx = max(maxx, x); maxy = max(maxy, y)
m = 26
im.crop((max(0, minx-m), max(0, miny-m), min(W, maxx+m), min(H, maxy+m))).save(out)
print("   →", out, Image.open(out).size)
PY
}

render_pdf() {
  echo "→ 输出 $NOTE_PDF"
  "$CHROME" --headless --disable-gpu --no-sandbox --no-pdf-header-footer \
    --user-data-dir="$PROFILE" --print-to-pdf="$NOTE_PDF" \
    "file://$PWD/_make_sensor2perception_note.html" >/dev/null 2>&1

  echo "→ 输出 $QA_PDF"
  "$CHROME" --headless --disable-gpu --no-sandbox --no-pdf-header-footer \
    --user-data-dir="$PROFILE" --print-to-pdf="$QA_PDF" \
    "file://$PWD/_make_sensor2perception_qa.html" >/dev/null 2>&1
}

case "$MODE" in
  fig) render_fig ;;
  pdf) render_pdf ;;
  all) render_fig; render_pdf ;;
  *) echo "用法：$0 [fig|pdf|all]"; exit 1 ;;
esac

echo "完成。"
