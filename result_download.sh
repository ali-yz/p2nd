#!/usr/bin/env bash
set -euo pipefail

ROOT="data/output"
OUTDIR="data/output_pngs"  # all prefixed PNG copies will go here
TS="$(date +%Y%m%d_%H%M%S)"
ARCHIVE="${OUTDIR}_${TS}.tar.gz"

echo "==> Cleaning up any existing *.png.gz under ${ROOT}..."
find "$ROOT" -type f -name '*.png.gz' -print -delete || true

echo "==> Preparing output folder: ${OUTDIR}"
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

echo "==> Collecting .png files and copying with folder-name prefix..."
# Find and copy with prefix from the *immediate* parent directory
# e.g., data/output/pc20_v3/phi.png -> data/output_pngs/pc20_v3_phi.png
while IFS= read -r -d '' f; do
  parent="$(basename "$(dirname "$f")")"
  base="$(basename "$f")"
  dest="${OUTDIR}/${parent}_${base}"

  # Avoid collisions just in case
  if [[ -e "$dest" ]]; then
    i=1
    while [[ -e "${OUTDIR}/${parent}_${i}_${base}" ]]; do
      ((i++))
    done
    dest="${OUTDIR}/${parent}_${i}_${base}"
  fi

  cp -p "$f" "$dest"
  echo "   + ${dest}"
done < <(find "$ROOT" -type f -name '*.png' -print0)

echo "==> Creating gzip-compressed tar archive: ${ARCHIVE}"
# tar with gzip (keeps original files intact)
tar -C "$(dirname "$OUTDIR")" -czf "$ARCHIVE" "$(basename "$OUTDIR")"

echo "==> Done."
echo "Folder bundled: ${OUTDIR}"
echo "Archive created: ${ARCHIVE}"
