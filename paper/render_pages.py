"""Rasterise main.pdf into page montages for a quick visual check."""
import sys
from pathlib import Path
import pymupdf
from PIL import Image

HERE = Path(__file__).resolve().parent
doc = pymupdf.open(HERE / "main.pdf")
zoom = float(sys.argv[1]) if len(sys.argv) > 1 else 0.9
pages = [p.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom)) for p in doc]
imgs = [Image.frombytes("RGB", (px.width, px.height), px.samples) for px in pages]
out = HERE / "preview"
out.mkdir(exist_ok=True)
for i, im in enumerate(imgs):
    im.save(out / f"page{i+1:02d}.png")
# montages of 3 pages per row
w, h = imgs[0].size
per = 3
for start in range(0, len(imgs), per * 2):
    chunk = imgs[start:start + per * 2]
    rows = (len(chunk) + per - 1) // per
    sheet = Image.new("RGB", (w * per, h * rows), "white")
    for k, im in enumerate(chunk):
        sheet.paste(im, ((k % per) * w, (k // per) * h))
    sheet.save(out / f"sheet_{start+1:02d}-{start+len(chunk):02d}.png")
print(len(imgs), "pages;", w, "x", h)
