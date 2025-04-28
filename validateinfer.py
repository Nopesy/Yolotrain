import os
import sys
import xml.etree.ElementTree as ET
from typing import List, Tuple

import cv2
import numpy as np
import math
import tensorflow as tf

# ← changed here: import your new loss
from yolo import yolo_loss  
from validatetrain import Darknet19, letterbox

# ─── CONFIG ─────────────────────────────────────────────────
ANNOT_DIR     = "dataset/VOCdevkit/VOC2012/Annotations"
IMAGE_DIR     = "dataset/VOCdevkit/VOC2012/JPEGImages"
IMAGESET_MAIN = "dataset/VOCdevkit/VOC2012/ImageSets/Main"
OUTPUT_DIR    = "detections/step3"
CLASS_NAMES   = [
    "aeroplane","bicycle","bird","boat","bottle","bus","car",
    "cat","chair","cow","diningtable","dog","horse","motorbike",
    "person","pottedplant","sheep","sofa","train","tvmonitor"
]
GRID_SIZE      = 13
INPUT_SIZE     = 416
CONF_THRESH    = 0.5
IOU_THRESH     = 0.5
MAX_DETECTIONS = 50
ANCHORS_PX: List[Tuple[float,float]] = [
    (55.4, 80.7), (139.2, 248.4), (327.9, 319.7)
]

os.makedirs(OUTPUT_DIR, exist_ok=True)
sigmoid = lambda x: 1.0 / (1.0 + np.exp(-x))


def load_gt(xml_path: str):
    """Parse VOC XML → (boxes, classes)"""
    root = ET.parse(xml_path).getroot()
    boxes, classes = [], []
    for obj in root.findall('object'):
        cls = CLASS_NAMES.index(obj.find('name').text)
        b = obj.find('bndbox')
        x1 = float(b.find('xmin').text)
        y1 = float(b.find('ymin').text)
        x2 = float(b.find('xmax').text)
        y2 = float(b.find('ymax').text)
        boxes.append([x1,y1,x2,y2])
        classes.append(cls)
    return np.array(boxes, dtype=np.float32), np.array(classes, dtype=np.int32)


def infer_and_visualize(xml_path: str, model):
    # --- 1) GT and best-anchor for debug
    gt_boxes, gt_cls = load_gt(xml_path)
    x1,y1,x2,y2 = gt_boxes[0]
    orig = cv2.imread(os.path.join(IMAGE_DIR, os.path.basename(xml_path)[:-4] + '.jpg'))
    lb_tmp, scale, pad_y, pad_x = letterbox(orig[...,::-1], target=INPUT_SIZE)
    xmin, ymin = x1*scale+pad_x, y1*scale+pad_y
    xmax, ymax = x2*scale+pad_x, y2*scale+pad_y
    bw, bh     = xmax-xmin, ymax-ymin
    cx, cy     = xmin + bw/2, ymin + bh/2
    x_rel, y_rel = cx/INPUT_SIZE, cy/INPUT_SIZE
    col = min(int(x_rel * GRID_SIZE), GRID_SIZE-1)
    row = min(int(y_rel * GRID_SIZE), GRID_SIZE-1)
    ious = []
    for aw, ah in ANCHORS_PX:
        inter = min(bw,aw) * min(bh,ah)
        uni   = bw*bh + aw*ah - inter
        ious.append(inter/uni)
    best_anchor = int(np.argmax(ious))

    print(f"\n>> IMAGE ID: {os.path.basename(xml_path)[:-4]}")
    print(f"   GT BBox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}), Class={CLASS_NAMES[gt_cls[0]]}")
    print(f"   GT center→ cell=(r{row},c{col}), best_anchor={best_anchor}, IoUs={['%.3f'%i for i in ious]}")

    # --- 2) load & preprocess
    img_id = os.path.basename(xml_path).split('.')[0]
    orig   = cv2.imread(os.path.join(IMAGE_DIR, img_id + '.jpg'))
    lb, scale, pad_y, pad_x = letterbox(orig[...,::-1], target=INPUT_SIZE)
    inp = lb[None,...]

    # --- 3) predict ---
    pred = model.predict(inp)[0]  # shape (13,13,3,5+20)

    cell = INPUT_SIZE / GRID_SIZE
    all_boxes, all_scores, all_classes = [], [], []

    # --- 4) decode & debug that one cell/anchor ---
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            for a, (aw, ah) in enumerate(ANCHORS_PX):
                tx, ty, tw, th, to = pred[i,j,a,:5]

                if (i,j,a)==(row,col,best_anchor):
                    print("\n   -- RAW PRED at (r,c,a) =",
                          f"({row},{col},{best_anchor}):",
                          f"tx={tx:.3f}, ty={ty:.3f}, tw={tw:.3f}, th={th:.3f}, to={to:.3f}")

                # --------  decode centre  --------
                sx, sy = sigmoid(tx), sigmoid(ty)
                cx = (j + sx) * cell
                cy = (i + sy) * cell

                # --------  decode w,h  ( **match training clamp** )  --------
                # same constants you used in yolo_loss
                # MAX_WH   = 2.0
                # clamp_tw = (2.0 / math.pi) * MAX_WH * math.atan(tw)
                # clamp_th = (2.0 / math.pi) * MAX_WH * math.atan(th)

                bw = math.exp(tw) * aw
                bh = math.exp(th) * ah

                x1p = (cx - bw/2 - pad_x) / scale
                y1p = (cy - bh/2 - pad_y) / scale
                x2p = (cx + bw/2 - pad_x) / scale
                y2p = (cy + bh/2 - pad_y) / scale

                conf        = sigmoid(to)
                class_probs = tf.nn.softmax(pred[i,j,a,5:]).numpy()
                cls_id      = int(np.argmax(class_probs))
                score       = conf * class_probs[cls_id]

                if (i,j,a)==(row,col,best_anchor):
                    print("   -- POST PRED: ",
                          f"cx={cx:.1f}, cy={cy:.1f}, bw={bw:.1f}, bh={bh:.1f}")
                    print("                  ",
                          f"mapped_box=({x1p:.1f},{y1p:.1f},{x2p:.1f},{y2p:.1f}),",
                          f"conf*cls_prob={score:.3f}, cls={CLASS_NAMES[cls_id]}")

                if score > CONF_THRESH:
                    all_boxes.append([x1p,y1p,x2p,y2p])
                    all_scores.append(score)
                    all_classes.append(cls_id)

    # --- 5) NMS + render ---
    if not all_boxes:
        print("   no detections above threshold")
        return

    sel = tf.image.non_max_suppression(
        tf.constant(all_boxes), tf.constant(all_scores),
        MAX_DETECTIONS, iou_threshold=IOU_THRESH, score_threshold=CONF_THRESH
    ).numpy()

    final_boxes   = np.array(all_boxes)[sel]
    final_scores  = np.array(all_scores)[sel]
    final_classes = np.array(all_classes)[sel]

    canvas = orig.copy()
    cv2.rectangle(canvas, (int(x1),int(y1)),(int(x2),int(y2)), (0,255,0),2)
    cv2.putText(canvas, CLASS_NAMES[gt_cls[0]], (int(x1),int(y1)-5),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
    for (x1p,y1p,x2p,y2p), s, c in zip(final_boxes, final_scores, final_classes):
        cv2.rectangle(canvas, (int(x1p),int(y1p)),(int(x2p),int(y2p)), (0,0,255),2)
        cv2.putText(canvas, f"{CLASS_NAMES[c]}:{s:.2f}",
                    (int(x1p),int(y1p)-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1)

    out = os.path.join(OUTPUT_DIR, img_id + "_debug.png")
    cv2.imwrite(out, canvas)
    print(f"   Saved visualization → {out}\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validateinfer.py <image_id>")
        sys.exit(1)

    image_id = sys.argv[1]
    xml_path = os.path.join(ANNOT_DIR, f"{image_id}.xml")
    if not os.path.exists(xml_path):
        raise FileNotFoundError(xml_path)

    # ← changed here: load with your new loss
    model = tf.keras.models.load_model("best.keras", compile=False)

    infer_and_visualize(xml_path, model)