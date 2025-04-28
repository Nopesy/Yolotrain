import os
import random
import glob
import math
import csv
import xml.etree.ElementTree as ET
from typing import List, Tuple

import cv2
import numpy as np
import tensorflow as tf

from validatetrain import Darknet19, letterbox
from yolo import yolo_loss

# ─── CONFIG ─────────────────────────────────────────────────
ANNOT_DIR   = "dataset/VOCdevkit/VOC2012/Annotations"
IMAGE_DIR   = "dataset/VOCdevkit/VOC2012/JPEGImages"
OUTPUT_DIR  = "Ablation/debug_detections_fixede3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = [
    "aeroplane","bicycle","bird","boat","bottle","bus","car",
    "cat","chair","cow","diningtable","dog","horse","motorbike",
    "person","pottedplant","sheep","sofa","train","tvmonitor"
]
GRID_SIZE   = 13
INPUT_SIZE  = 416
CONF_THRESH = 1.0
ANCHORS_PX: List[Tuple[float,float]] = [
    (55.4,  80.7),
    (139.2, 248.4),
    (327.9, 319.7),
]
MAX_WH = 2.0
sigmoid = lambda x: 1.0/(1.0+np.exp(-x))

# ─── MODEL ───────────────────────────────────────────────────
model = tf.keras.models.load_model(
    "fixede3.keras",
    compile=False,
    custom_objects={'yolo_loss': yolo_loss}
)

# ─── UTILITY FUNCTIONS ───────────────────────────────────────
def load_gt(xml_path: str):
    root = ET.parse(xml_path).getroot()
    boxes, classes = [], []
    for obj in root.findall("object"):
        cls = CLASS_NAMES.index(obj.find("name").text)
        b = obj.find("bndbox")
        xmin, ymin = float(b.find("xmin").text), float(b.find("ymin").text)
        xmax, ymax = float(b.find("xmax").text), float(b.find("ymax").text)
        boxes.append([xmin, ymin, xmax, ymax])
        classes.append(cls)
    return np.array(boxes, np.float32), np.array(classes, np.int32)

def decode_anchor(raw, col, row, a_idx):
    tx, ty, tw, th, to = raw[:5]
    conf = sigmoid(to)
    probs = tf.nn.softmax(raw[5:]).numpy()
    cls_id = int(np.argmax(probs))
    cell = INPUT_SIZE / GRID_SIZE
    sx, sy = sigmoid(tx), sigmoid(ty)
    cx, cy = (col + sx) * cell, (row + sy) * cell
    clamp_tw = (2.0/math.pi) * MAX_WH * math.atan(tw)
    clamp_th = (2.0/math.pi) * MAX_WH * math.atan(th)
    bw = math.exp(clamp_tw) * ANCHORS_PX[a_idx][0]
    bh = math.exp(clamp_th) * ANCHORS_PX[a_idx][1]
    x1, y1 = cx - bw/2, cy - bh/2
    x2, y2 = cx + bw/2, cy + bh/2
    return (x1, y1, x2, y2, conf, cls_id)

def iou(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    iw = max(0, min(ax2,bx2) - max(ax1,bx1))
    ih = max(0, min(ay2,by2) - max(ay1,by1))
    inter = iw * ih
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter/union if union > 0 else 0.0

def process_split(xml_list: List[str], csv_path: str):
    cell = INPUT_SIZE / GRID_SIZE
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'img_id','gt_class','row','col','correct_cls',
            'conf_best','iou_best','extra_fp_cnt'
        ])
        total = len(xml_list)
        for idx, xml in enumerate(xml_list, 1):
            img_id = os.path.basename(xml).split('.')[0]
            print(f"[{idx}/{total}] Processing {img_id}", flush=True)

            gt_boxes, gt_classes = load_gt(xml)
            orig = cv2.imread(os.path.join(IMAGE_DIR, img_id + '.jpg'))
            rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
            lb, scale, pad_y, pad_x = letterbox(rgb, target=INPUT_SIZE)
            pred = model.predict(lb[None,...], verbose=0)[0]

            valid_cells = set()
            resp = []
            for (xmin,ymin,xmax,ymax) in gt_boxes:
                bx = (xmin + xmax)/2 * scale + pad_x
                by = (ymin + ymax)/2 * scale + pad_y
                col = min(int(bx/cell), GRID_SIZE-1)
                row = min(int(by/cell), GRID_SIZE-1)
                valid_cells.add((row, col))
                resp.append((row, col))

            extra_fp = 0
            for r in range(GRID_SIZE):
                for c in range(GRID_SIZE):
                    for a in range(len(ANCHORS_PX)):
                        _,_,_,_,conf, _ = decode_anchor(pred[r,c,a], c, r, a)
                        if conf >= CONF_THRESH and (r,c) not in valid_cells:
                            extra_fp += 1

            for (row,col),(xmin,ymin,xmax,ymax),cid in zip(resp, gt_boxes, gt_classes):
                best_conf, best_iou, correct = 0.0, 0.0, 0
                gt_scaled = (
                    xmin*scale+pad_x, ymin*scale+pad_y,
                    xmax*scale+pad_x, ymax*scale+pad_y
                )
                for a in range(len(ANCHORS_PX)):
                    x1,y1,x2,y2,conf,cls_id = decode_anchor(pred[row,col,a], col, row, a)
                    if cls_id == cid and conf > best_conf:
                        best_conf = conf
                        best_iou  = iou((x1,y1,x2,y2), gt_scaled)
                        correct   = 1
                writer.writerow([
                    img_id,
                    CLASS_NAMES[cid],
                    row,
                    col,
                    correct,
                    f"{best_conf:.4f}",
                    f"{best_iou:.4f}",
                    extra_fp
                ])

if __name__ == '__main__':
    all_xmls  = sorted(glob.glob(os.path.join(ANNOT_DIR, '*.xml')))
    split_idx = int(0.8 * len(all_xmls))
    train_xmls = all_xmls[:split_idx]
    val_xmls   = all_xmls[split_idx:]

    print('Processing training split...')
    process_split(train_xmls, os.path.join(OUTPUT_DIR, 'train_metrics.csv'))
    print('Processing validation split...')
    process_split(val_xmls, os.path.join(OUTPUT_DIR, 'val_metrics.csv'))
    print('Done')