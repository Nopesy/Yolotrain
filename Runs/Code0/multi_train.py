import os
import glob
import random
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
import cv2
from Code0.multi_yolo import Darknet19, yolo_loss

# ——— CONFIG ———
ANNOT_DIR        = "dataset/VOCdevkit/VOC2012/Annotations"
IMAGE_DIR        = "dataset/VOCdevkit/VOC2012/JPEGImages"
CLASS_NAMES      = ["aeroplane","bicycle","bird","boat","bottle",
                    "bus","car","cat","chair","cow","diningtable",
                    "dog","horse","motorbike","person","pottedplant",
                    "sheep","sofa","train","tvmonitor"]
NUM_CLASSES      = len(CLASS_NAMES)
INPUT_SIZE       = 416
GRID_SIZE        = 13
BATCH_SIZE       = 8
EPOCHS           = 5
# Optional model init
MODEL_INIT_PATH  = "yolo_anchors_single_scale.h5"  # set to None to train from scratch

# ——— load anchors ———
ANCHORS = [
    (55.4, 80.7),
    (139.2, 248.4),
    (327.9, 319.7),
]
B = len(ANCHORS)

# ——— helper to pick best anchor by wh difference ———
def best_anchor_idx(w_cell, h_cell):
    diffs = [abs(w_cell - aw) + abs(h_cell - ah) for aw, ah in ANCHORS]
    return int(np.argmin(diffs))

# ——— parse + encode one example ———
def parse_example(xml_path_str):
    xml_bytes = xml_path_str.numpy() if hasattr(xml_path_str, 'numpy') else xml_path_str
    xml_path = xml_bytes.decode('utf-8') if isinstance(xml_bytes, (bytes, bytearray)) else xml_bytes
    tree = ET.parse(xml_path)
    root = tree.getroot()
    fname = root.find('filename').text

    # load and letterbox image
    img = cv2.imread(os.path.join(IMAGE_DIR, fname))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h0, w0, _ = img.shape
    scale = min(INPUT_SIZE/w0, INPUT_SIZE/h0)
    nw, nh = int(w0*scale), int(h0*scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((INPUT_SIZE,INPUT_SIZE,3), 128, np.uint8)
    dx, dy = (INPUT_SIZE-nw)//2, (INPUT_SIZE-nh)//2
    canvas[dy:dy+nh, dx:dx+nw] = resized
    inp = canvas.astype(np.float32) / 255.0

    # prepare y_true
    y_true = np.zeros((GRID_SIZE, GRID_SIZE, B, 5+NUM_CLASSES), dtype=np.float32)
    for obj in root.findall('object'):
        cls = obj.find('name').text
        cid = CLASS_NAMES.index(cls)
        b = obj.find('bndbox')
        xmin, ymin, xmax, ymax = [float(b.find(tag).text) for tag in ('xmin','ymin','xmax','ymax')]
        xmin, ymin = xmin*scale + dx, ymin*scale + dy
        xmax, ymax = xmax*scale + dx, ymax*scale + dy
        bw, bh = xmax-xmin, ymax-ymin
        xc, yc = xmin + bw/2, ymin + bh/2
        x_norm, y_norm = xc/INPUT_SIZE, yc/INPUT_SIZE
        col, row = int(x_norm*GRID_SIZE), int(y_norm*GRID_SIZE)
        tx = x_norm*GRID_SIZE - col
        ty = y_norm*GRID_SIZE - row
        w_cell = bw / (INPUT_SIZE/GRID_SIZE)
        h_cell = bh / (INPUT_SIZE/GRID_SIZE)
        a_idx = best_anchor_idx(w_cell, h_cell)
        y_true[row, col, a_idx, :2] = [tx, ty]
        y_true[row, col, a_idx, 2:4] = [w_cell, h_cell]
        y_true[row, col, a_idx, 4] = 1.0
        y_true[row, col, a_idx, 5+cid] = 1.0

    return inp, y_true

# ——— tf.data pipeline ———
all_xmls = sorted(glob.glob(os.path.join(ANNOT_DIR, '*.xml')))
random.shuffle(all_xmls)
split = int(0.8 * len(all_xmls))
train_paths, val_paths = all_xmls[:split], all_xmls[split:]

@tf.function
def load_and_preprocess(xml_path):
    inp, y = tf.py_function(
        func=lambda p: parse_example(p),
        inp=[xml_path],
        Tout=(tf.float32, tf.float32)
    )
    inp.set_shape((INPUT_SIZE, INPUT_SIZE, 3))
    y.set_shape((GRID_SIZE, GRID_SIZE, B, 5+NUM_CLASSES))
    return inp, y

train_ds = tf.data.Dataset.from_tensor_slices(train_paths)
train_ds = (train_ds.shuffle(len(train_paths))
                     .map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                     .batch(BATCH_SIZE, drop_remainder=True)
                     .prefetch(tf.data.AUTOTUNE))

val_ds = tf.data.Dataset.from_tensor_slices(val_paths)
val_ds = (val_ds.shuffle(len(val_paths))
                    .map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                    .batch(BATCH_SIZE, drop_remainder=True)
                    .prefetch(tf.data.AUTOTUNE))

# ——— build/ load model & train ———
if MODEL_INIT_PATH and os.path.exists(MODEL_INIT_PATH):
    print(f"Loading existing model from {MODEL_INIT_PATH}")
    model = tf.keras.models.load_model(
        MODEL_INIT_PATH,
        custom_objects={'yolo_loss': yolo_loss}
    )
else:
    model = Darknet19(input_shape=(INPUT_SIZE,INPUT_SIZE,3),
                      num_classes=NUM_CLASSES,
                      anchors=ANCHORS)
    model.compile(optimizer='adam', loss=yolo_loss)

model.summary()
model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
model.save(MODEL_INIT_PATH or 'yolo_anchors_single_scale.h5')
