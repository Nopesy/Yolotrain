import os
import random
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ——— Update these to your local VOC paths ———
ANNOT_DIR = "dataset/VOCdevkit/VOC2012/Annotations"
IMAGE_DIR      = "dataset/VOCdevkit/VOC2012/JPEGImages"

# Pick a random XML annotation
xmls = [f for f in os.listdir(ANNOT_DIR) if f.endswith('.xml')]
xml_file = random.choice(xmls)
xml_path = os.path.join(ANNOT_DIR, xml_file)

# Parse XML
tree = ET.parse(xml_path)
root = tree.getroot()
size = root.find('size')
W, H = int(size.find('width').text), int(size.find('height').text)
img_name = root.find('filename').text
img_path = os.path.join(IMAGE_DIR, img_name)

# Load image
img_bgr = cv2.imread(img_path)
if img_bgr is None:
    raise IOError(f"Could not load {img_path}")
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Grid parameters
S = 13
cell_w, cell_h = W / S, H / S

# Begin plotting
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(img)
ax.set_title(f"{img_name} — all objects, cells & Δx/Δy")

# Draw the 13×13 grid
for i in range(1, S):
    ax.axvline(i * cell_w, color='white', linestyle='--', linewidth=0.5)
    ax.axhline(i * cell_h, color='white', linestyle='--', linewidth=0.5)

# Process every object
for obj in root.findall('object'):
    cls = obj.find('name').text
    bb = obj.find('bndbox')
    xmin = int(bb.find('xmin').text)
    ymin = int(bb.find('ymin').text)
    xmax = int(bb.find('xmax').text)
    ymax = int(bb.find('ymax').text)
    bw, bh = xmax - xmin, ymax - ymin

    # Draw bounding box + size label
    rect = patches.Rectangle((xmin, ymin), bw, bh,
                             linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)
    ax.text(xmin, ymin - 12, f"{cls} ({bw}×{bh})",
            color='red', fontsize=8, backgroundcolor='white')

    # Compute center (absolute) and normalized over entire image
    xc, yc = xmin + bw/2, ymin + bh/2
    xn, yn = xc / W, yc / H

    # Determine cell indices
    cell_i = int(xn * S)
    cell_j = int(yn * S)

    # Compute within‐cell offsets
    t_x = xn * S - cell_i
    t_y = yn * S - cell_j

    # Pixel arrow lengths within the cell
    dx = t_x * cell_w
    dy = t_y * cell_h

    # Highlight the responsible cell
    cell_x0 = cell_i * cell_w
    cell_y0 = cell_j * cell_h
    cell_rect = patches.Rectangle((cell_x0, cell_y0),
                                  cell_w, cell_h,
                                  linewidth=1.5,
                                  edgecolor='yellow',
                                  facecolor='none')
    ax.add_patch(cell_rect)

    # Draw Δx arrow from cell top‑left
    ax.annotate("",
                xy=(cell_x0 + dx, cell_y0),
                xytext=(cell_x0, cell_y0),
                arrowprops=dict(arrowstyle="->", color="cyan", lw=1.5))
    ax.text(cell_x0 + dx/2, cell_y0 - 8,
            f"Δx={t_x:.2f}",
            color='cyan', fontsize=7,
            ha='center', va='bottom',
            backgroundcolor='black')

    # Draw Δy arrow downward from end of Δx
    ax.annotate("",
                xy=(cell_x0 + dx, cell_y0 + dy),
                xytext=(cell_x0 + dx, cell_y0),
                arrowprops=dict(arrowstyle="->", color="magenta", lw=1.5))
    ax.text(cell_x0 + dx + 4, cell_y0 + dy/2,
            f"Δy={t_y:.2f}",
            color='magenta', fontsize=7,
            ha='left', va='center',
            backgroundcolor='black')

ax.axis('off')
plt.tight_layout()
plt.show()