import os, glob, random, xml.etree.ElementTree as ET
from typing import List, Tuple, Dict

import cv2
import numpy as np
import tensorflow as tf
from functools import partial


# for printing inside tf
tf.config.optimizer.set_jit(False)

from yolo import Darknet19, yolo_loss, yolo_loss_debug

# ──────────────────── CONFIG ────────────────────
ANNOT_DIR       = "dataset/VOCdevkit/VOC2012/Annotations"
IMAGE_DIR       = "dataset/VOCdevkit/VOC2012/JPEGImages"
IMAGESET_MAIN   = "dataset/VOCdevkit/VOC2012/ImageSets/Main"
CLASS_NAMES     = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
NUM_CLASSES     = len(CLASS_NAMES)
IMG_SIZE        = 416          # input to the network (square)
BATCH_SIZE      = 1            # one image per batch for Phase 1
EPOCHS          = 1000         # few epochs to verify loss trend
# set seeds

ANCHORS_PX: List[Tuple[float, float]] = [
    (55.4, 80.7), (139.2, 248.4), (327.9, 319.7)
]
NUM_ANCHORS     = len(ANCHORS_PX)
MODEL_INIT_PATH = None         # set to "best.h5" to resume
CSV_LOG_FILE    = "batch1_loss.csv"

# ───────────────── DATA AUGMENTATION ─────────────────

def jitter_image(img: np.ndarray, boxes: List[Tuple[float, float, float, float]],
                 max_jitter: float) -> Tuple[np.ndarray, List[Tuple[float,float,float,float]]]:
    """Random aspect-ratio + scale jitter"""
    if max_jitter == 0:
        return img, boxes
    h, w = img.shape[:2]
    jx = random.uniform(1 - max_jitter, 1 + max_jitter)
    jy = random.uniform(1 - max_jitter, 1 + max_jitter)
    img = cv2.resize(img, (int(w * jx), int(h * jy)))
    boxes = [(xmin * jx, ymin * jy, xmax * jx, ymax * jy) for xmin, ymin, xmax, ymax in boxes]
    return img, boxes


def hflip(img: np.ndarray, boxes: List[Tuple[float, float, float, float]],
          enabled: bool) -> Tuple[np.ndarray, List[Tuple[float,float,float,float]]]:
    if not enabled or random.random() > 0.5:
        return img, boxes
    img = img[:, ::-1]
    w = img.shape[1]
    boxes = [(w - xmax, ymin, w - xmin, ymax) for xmin, ymin, xmax, ymax in boxes]
    return img, boxes


def color_jitter(img: np.ndarray, brightness: float, contrast: float, saturation: float) -> np.ndarray:
    if brightness == contrast == saturation == 0:
        return img
    tf_img = tf.convert_to_tensor(img)
    tf_img = tf.image.random_brightness(tf_img, brightness)
    tf_img = tf.image.random_contrast(tf_img, 1 - contrast, 1 + contrast)
    tf_img = tf.image.random_saturation(tf_img, 1 - saturation, 1 + saturation)
    tf_img = tf.clip_by_value(tf_img, 0, 255)
    return tf_img.numpy().astype(np.uint8)


def letterbox(img: np.ndarray, target: int = IMG_SIZE) -> Tuple[np.ndarray, float, int, int]:
    """Resize and pad to square keeping aspect ratio"""
    h, w = img.shape[:2]
    scale = min(target / w, target / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((target, target, 3), 128, np.uint8)
    pad_x, pad_y = (target - nw) // 2, (target - nh) // 2
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized
    return canvas.astype(np.float32) / 255.0, scale, pad_y, pad_x

# ───────────────── XML → TENSOR ─────────────────

def parse_xml(xml_path_tensor: tf.Tensor,
              max_jitter: tf.float32, flip: tf.bool,
              brightness: tf.float32, contrast: tf.float32, saturation: tf.float32):
    """Inside tf.py_function → NumPy/CPU code."""
    xml_path = xml_path_tensor.numpy().decode()
    root     = ET.parse(xml_path).getroot()

    # 1) load raw boxes & labels
    boxes, labels = [], []
    for obj in root.findall("object"):
        labels.append(obj.find("name").text)
        b = obj.find("bndbox")
        boxes.append((
            float(b.find("xmin").text),
            float(b.find("ymin").text),
            float(b.find("xmax").text),
            float(b.find("ymax").text),
        ))

    # 2) load & augment image
    img = cv2.cvtColor(
        cv2.imread(os.path.join(IMAGE_DIR, root.find("filename").text)),
        cv2.COLOR_BGR2RGB
    )
    img, boxes = jitter_image(img, boxes, float(max_jitter))
    img, boxes = hflip(img, boxes, bool(flip))
    img         = color_jitter(img, float(brightness),
                               float(contrast), float(saturation))

    # 3) letterbox → normalized float32 in [0,1]
    img, scale, pad_y, pad_x = letterbox(img)
    gsize = img.shape[0] // 32  # e.g. 13

    # 4) prepare target tensor
    y_true = np.zeros((gsize, gsize, NUM_ANCHORS, 5 + NUM_CLASSES),
                      np.float32)

    for (xmin, ymin, xmax, ymax), name in zip(boxes, labels):
        cid = CLASS_NAMES.index(name)

        # map GT box to letterbox coords
        xmin = xmin * scale + pad_x
        ymin = ymin * scale + pad_y
        xmax = xmax * scale + pad_x
        ymax = ymax * scale + pad_y

        bw = xmax - xmin
        bh = ymax - ymin
        xc = xmin + bw * 0.5
        yc = ymin + bh * 0.5

        # pick best anchor by IoU in pixels
        ious = [
            (min(bw, aw) * min(bh, ah)) /
            (bw * bh + aw * ah -
             min(bw, aw) * min(bh, ah) + 1e-9)
            for aw, ah in ANCHORS_PX
        ]
        a = int(np.argmax(ious))

        # grid cell coords + offsets
        col = min(int(xc / IMG_SIZE * gsize), gsize - 1)
        row = min(int(yc / IMG_SIZE * gsize), gsize - 1)
        dx  = xc / IMG_SIZE * gsize - col   # ∈ [0,1]
        dy  = yc / IMG_SIZE * gsize - row

        # compute log-ratio targets (tw, th) in *cell* units
        #  - add eps inside ratio to avoid log(0)
        bw_cells = bw / (IMG_SIZE / gsize) + 1e-9
        bh_cells = bh / (IMG_SIZE / gsize) + 1e-9
        aw_cells = ANCHORS_PX[a][0] / (IMG_SIZE / gsize) + 1e-9
        ah_cells = ANCHORS_PX[a][1] / (IMG_SIZE / gsize) + 1e-9

        tw = np.log(bw_cells / aw_cells)
        th = np.log(bh_cells / ah_cells)

        # 5) write into y_true as [dx, dy, tw, th]
        y_true[row, col, a, 0:4] = [dx, dy, tw, th]
        y_true[row, col, a, 4]   = 1.0               # objectness
        y_true[row, col, a, 5+cid] = 1.0             # one-hot class

    return img, y_true

# ───────────────── DATASET BUILDER ─────────────────

def build_dataset(xml_paths: List[str], *,
                  max_jitter=0.0, flip=False,
                  brightness=0.0, contrast=0.0, saturation=0.0):
    def wrapper(xml_path):
        img, y = tf.py_function(
            func=parse_xml,
            inp=[xml_path, max_jitter, flip, brightness, contrast, saturation],
            Tout=(tf.float32, tf.float32)
        )
        img.set_shape((IMG_SIZE, IMG_SIZE, 3))
        grid = IMG_SIZE // 32
        y.set_shape((grid, grid, NUM_ANCHORS, 5 + NUM_CLASSES))
        return img, y

    ds = (tf.data.Dataset.from_tensor_slices(xml_paths)
            .shuffle(len(xml_paths))
            .map(wrapper, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE))
    return ds

# ───────────────── CALLBACKS ─────────────────

class CSVLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_every=1, path=CSV_LOG_FILE):
        super().__init__()
        self.k = log_every
        self.f = open(path, "w"); self.f.write("epoch,batch,loss,val_loss\n")
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch + 1
    def on_train_batch_end(self, batch, logs=None):
        if batch % self.k == 0:
            self.f.write(f"{self.epoch},{batch},{logs['loss']:.6f},\n"); self.f.flush()
    def on_epoch_end(self, epoch, logs=None):
        self.f.write(f"{self.epoch},end,{logs['loss']:.6f},{logs.get('val_loss',np.nan):.6f}\n"); self.f.flush()
    def on_train_end(self, logs=None):
        self.f.close()

class SaveBestAtEnd(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.best = np.inf
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        val = logs.get("val_loss")
        if val is not None and val < self.best:
            self.best = val
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            # load best weights & save once
            self.model.set_weights(self.best_weights)
            self.model.save("best.keras")
            print(f"\nSaved best model (val_loss={self.best:.4f}) at end of training.")

# ───────────────── MAIN ─────────────────

def main():
    # # pick one positive example per class
    # selected_ids = []
    # for cls in CLASS_NAMES:
    #     txt_path = os.path.join(IMAGESET_MAIN, f"{cls}_trainval.txt")
    #     with open(txt_path) as f:
    #         for line in f:
    #             img_id, label = line.strip().split()
    #             if label == "1":
    #                 selected_ids.append(img_id)
    #                 break

    # # remove duplicates while preserving order
    # seen = set()
    # unique_ids = []
    # for img_id in selected_ids:
    #     if img_id not in seen:
    #         seen.add(img_id)
    #         unique_ids.append(img_id)

    # # build xml list and use same for train & val
    # xml_files = [os.path.join(ANNOT_DIR, f"{img_id}.xml") for img_id in unique_ids]

    # print out the 20 image IDs used for training/validation
    # for img_id in unique_ids:
    #     print(img_id)

    # train_xml = val_xml = xml_files
    xml_files = [os.path.join(ANNOT_DIR, "2007_000243.xml")]
    train_xml = val_xml = xml_files
    
    train_ds = build_dataset(train_xml)
    val_ds   = build_dataset(val_xml)

    def yolo_loss_with_step(y_true, y_pred):
            # closes over the `optimizer` name above
        return yolo_loss(
            y_true=y_true,
            y_pred=y_pred,
            lambda_coord=2.0,
            lambda_noobj=0.1,
        )

    model = Darknet19(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        num_classes=NUM_CLASSES,
        B=NUM_ANCHORS
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss=yolo_loss_with_step, run_eagerly=True)      # ← this will let tf.print work normally

    if MODEL_INIT_PATH and os.path.exists(MODEL_INIT_PATH):
        print(f"▶ Restoring weights from {MODEL_INIT_PATH}")
        model.load_weights(MODEL_INIT_PATH)

    callbacks = [
        CSVLogger(log_every=1),
        SaveBestAtEnd(),
        # tf.keras.callbacks.ModelCheckpoint(
        #     'best.keras', monitor='val_loss',
        #     save_best_only=True, verbose=1
        # )
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

if __name__ == "__main__":
    main()
