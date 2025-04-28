import os
import random
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ——— Configure these to your VOC2012 paths ———
ANNOT_DIR = "dataset/VOCdevkit/VOC2012/Annotations"
IMAGE_DIR = "dataset/VOCdevkit/VOC2012/JPEGImages"

# ---------- Utility Functions ----------

def parse_annotations(xml_path):
    """Parse VOC XML, return filename, image size, list of boxes and labels."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    W = int(root.find('size/width').text)
    H = int(root.find('size/height').text)
    filename = root.find('filename').text
    boxes, labels = [], []
    for obj in root.findall('object'):
        labels.append(obj.find('name').text)
        b = obj.find('bndbox')
        xmin = int(b.find('xmin').text)
        ymin = int(b.find('ymin').text)
        xmax = int(b.find('xmax').text)
        ymax = int(b.find('ymax').text)
        boxes.append((xmin, ymin, xmax, ymax))
    return filename, (W, H), boxes, labels

def compute_metrics(box, img_size, S=13):
    """
    Given box=(xmin,ymin,xmax,ymax), img_size=(W,H), compute:
      cell_i, cell_j, t_x, t_y, t_w, t_h
    """
    xmin, ymin, xmax, ymax = box
    W, H = img_size
    bw, bh = xmax - xmin, ymax - ymin
    xc, yc = xmin + bw/2, ymin + bh/2
    xn, yn = xc / W, yc / H
    cell_i = int(xn * S)
    cell_j = int(yn * S)
    t_x = xn * S - cell_i
    t_y = yn * S - cell_j
    cell_w, cell_h = W / S, H / S
    t_w = bw / cell_w
    t_h = bh / cell_h
    return cell_i, cell_j, t_x, t_y, t_w, t_h

def letterbox_image(image, target_size=(416, 416)):
    tw, th = target_size
    h, w = image.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((th, tw, 3), 128, dtype=np.uint8)
    dx, dy = (tw - nw) // 2, (th - nh) // 2
    canvas[dy:dy+nh, dx:dx+nw] = resized
    return canvas, scale, dy, dx

def transform_bbox_to_letterbox(box, scale, dx, dy):
    x1,y1,x2,y2 = box
    return (x1*scale+dx, y1*scale+dy, x2*scale+dx, y2*scale+dy)

def inverse_transform_bbox(box, scale, dx, dy):
    x1,y1,x2,y2 = box
    return ((x1-dx)/scale, (y1-dy)/scale, (x2-dx)/scale, (y2-dy)/scale)

# ---------- Main Script ----------

# 1) Pick a random annotation
xml_files = [f for f in os.listdir(ANNOT_DIR) if f.endswith('.xml')]
xml_file = random.choice(xml_files)
xml_path = os.path.join(ANNOT_DIR, xml_file)

# 2) Parse and load image
filename, (W, H), boxes, labels = parse_annotations(xml_path)
img = cv2.imread(os.path.join(IMAGE_DIR, filename))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 3) Compute original metrics
orig_metrics = [compute_metrics(b, (W, H)) for b in boxes]

# 4) Letterbox image & transform boxes
lb_img, scale, dy, dx = letterbox_image(img, (416, 416))
lb_boxes = [transform_bbox_to_letterbox(b, scale, dx, dy) for b in boxes]

# 5) Compute metrics on letterboxed image
lb_metrics = [compute_metrics(b, (416, 416)) for b in lb_boxes]

# 6) Inverse transform and compute recovered metrics
rec_boxes = [inverse_transform_bbox(b, scale, dx, dy) for b in lb_boxes]
rec_metrics = [compute_metrics(b, (W, H)) for b in rec_boxes]

# 7) Print metrics comparison
print(f"Image: {filename}  Size: {W}×{H}\n")
for i, lbl in enumerate(labels):
    o = orig_metrics[i]
    l = lb_metrics[i]
    r = rec_metrics[i]
    print(f"{lbl}:")
    print(f"  Original:       cell=({o[1]},{o[0]})  tx={o[2]:.3f}  ty={o[3]:.3f}  tw={o[4]:.3f}  th={o[5]:.3f}")
    print(f"  Letterbox:      cell=({l[1]},{l[0]})  tx={l[2]:.3f}  ty={l[3]:.3f}  tw={l[4]:.3f}  th={l[5]:.3f}")
    print(f"  Recovered:      cell=({r[1]},{r[0]})  tx={r[2]:.3f}  ty={r[3]:.3f}  tw={r[4]:.3f}  th={r[5]:.3f}\n")

# 8) Visualize results
fig, axes = plt.subplots(1, 3, figsize=(18,6))
titles = ["Original", "Letterboxed", "Recovered"]
images = [img, lb_img, img]
boxes_list = [boxes, lb_boxes, rec_boxes]
colors = ['red','blue','green']

for ax, im, blist, col, title in zip(axes, images, boxes_list, colors, titles):
    ax.imshow(im)
    ax.set_title(title)
    for b, lbl in zip(blist, labels):
        x1,y1,x2,y2 = b
        rect = patches.Rectangle((x1,y1), x2-x1, y2-y1,
                                 linewidth=2, edgecolor=col, facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1-6, lbl, color=col,
                fontsize=10, backgroundcolor='white')
    ax.axis('off')

plt.tight_layout()
plt.show()