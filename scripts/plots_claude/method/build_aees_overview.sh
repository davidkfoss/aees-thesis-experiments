#!/usr/bin/env bash
# Build the AEES overview TikZ figure.
#
# Compiles aees_overview.tex with pdflatex, tightly crops the resulting
# PDF to the figure's ink bounding box via ghostscript, then renders a
# high-resolution PNG companion via ImageMagick.
#
# Outputs (in results/plots_claude/method/):
#   - aees_overview.pdf            (vector, tightly cropped, for thesis)
#   - aees_overview.png            (raster, 220 dpi, for previews)
#   - aees_overview.summary.txt    (label inventory, for QA)
#
# Requires: pdflatex (TeX Live), gs, magick (ImageMagick 7).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
SRC="$HERE/aees_overview.tex"
OUT="$REPO/results/plots_claude/method"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

mkdir -p "$OUT"

# 1. Compile (run twice to settle any internal references; cheap).
pdflatex -interaction=nonstopmode -halt-on-error \
    -output-directory "$TMP" "$SRC" >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error \
    -output-directory "$TMP" "$SRC" >/dev/null

# 2. Crop the PDF to its ink bounding box plus a small uniform padding.
BBOX=$(gs -dNOPAUSE -dBATCH -sDEVICE=bbox "$TMP/aees_overview.pdf" 2>&1 \
       | grep HiResBoundingBox)
X1=$(awk '{print $2}' <<<"$BBOX")
Y1=$(awk '{print $3}' <<<"$BBOX")
X2=$(awk '{print $4}' <<<"$BBOX")
Y2=$(awk '{print $5}' <<<"$BBOX")
PAD=4
W=$(python3 -c "print(($X2 - $X1) + 2*$PAD)")
H=$(python3 -c "print(($Y2 - $Y1) + 2*$PAD)")
OX=$(python3 -c "print($PAD - $X1)")
OY=$(python3 -c "print($PAD - $Y1)")

gs -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -o "$OUT/aees_overview.pdf" \
   -dDEVICEWIDTHPOINTS=$W -dDEVICEHEIGHTPOINTS=$H -dFIXEDMEDIA \
   -c "<</PageOffset [$OX $OY]>> setpagedevice" \
   -f "$TMP/aees_overview.pdf"

# 3. Render PNG companion (220 dpi, alpha-flattened to white).
magick -density 220 "$OUT/aees_overview.pdf" \
       -background white -alpha remove -alpha off \
       "$OUT/aees_overview.png"

# 4. Refresh summary (label inventory matches the figure spec).
cat >"$OUT/aees_overview.summary.txt" <<'EOF'
schematic - no input data
source: scripts/plots_claude/method/aees_overview.tex (TikZ)
labels rendered:
  - Mini-batch
  - Forward / backward
  - Base optimizer
  - Parameter update
  - Episodic Bandit Controller (LR-multiplier)
  - Episodic Bandit Controller (Gradient noise)
  - Selected arm values
  - Reward $r_e$ from EMA loss
outputs:
  - results/plots_claude/method/aees_overview.pdf
  - results/plots_claude/method/aees_overview.png
EOF

echo "wrote $OUT/aees_overview.pdf"
echo "wrote $OUT/aees_overview.png"
echo "wrote $OUT/aees_overview.summary.txt"
