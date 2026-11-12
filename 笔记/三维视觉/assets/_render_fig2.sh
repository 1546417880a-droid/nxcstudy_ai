#!/bin/bash
# 重出「图 2 · 五步流水线」：HTML → Chrome 无头截图 → 裁到内容边界。
# 依赖：Google Chrome、Pillow（python3 -c "import PIL" 能通过）。
# 用法：./_render_fig2.sh            输出覆盖 ./自动驾驶-传感器链路-图2-五步流水线.png
set -euo pipefail
cd "$(dirname "$0")"

CHROME="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
PYTHON="${PYTHON:-python3}"
RAW="${TMPDIR:-/tmp}/fig2_sensor_chain_raw.png"

"$CHROME" --headless --disable-gpu --hide-scrollbars --no-sandbox \
  --disable-breakpad --disable-crash-reporter --no-first-run \
  --user-data-dir="${TMPDIR:-/tmp}/chrome-fig2-profile" \
  --force-device-scale-factor=2 --window-size=1560,1300 \
  --screenshot="$RAW" "file://$PWD/_make_sensor_chain_fig2.html" >/dev/null 2>&1

"$PYTHON" - "$RAW" <<'PY'
import sys
from PIL import Image

raw, out = sys.argv[1], "自动驾驶-传感器链路-图2-五步流水线.png"
im = Image.open(raw).convert("RGB")
bg = (247, 249, 252)                       # 页面底色 #F7F9FC
px = im.load()
W, H = im.size

def diff(c):
    return abs(c[0] - bg[0]) + abs(c[1] - bg[1]) + abs(c[2] - bg[2]) > 6

ys = [y for y in range(H) if any(diff(px[x, y]) for x in range(0, W, 4))]
top, bot = min(ys), max(ys)

# 宽度保持整幅（左右各 80 px 留白），上下各补 80 px
im.crop((0, max(0, top - 80), W, min(H, bot + 80))).save(out)
print("wrote", out, Image.open(out).size)
PY
