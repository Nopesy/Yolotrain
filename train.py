import os, glob, random, time, datetime, xml.etree.ElementTree as ET
from typing import List, Tuple, Dict

import cv2
import numpy as np
import tensorflow as tf

from yolo import Darknet19, yolo_loss

# ──────────────────── CONFIG ────────────────────
ANNOT_DIR       = "dataset/VOCdevkit/VOC2012/Annotations"
IMAGE_DIR       = "dataset/VOCdevkit/VOC2012/JPEGImages"
CLASS_NAMES     = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
    "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
NUM_CLASSES     = len(CLASS_NAMES)
IMG_SIZE        = 416          # input to the network (square)
BATCH_SIZE      = 8
EPOCHS          = 60

ANCHORS_PX: List[Tuple[float, float]] = [
    (55.4, 80.7), (139.2, 248.4), (327.9, 319.7)
]
NUM_ANCHORS     = len(ANCHORS_PX)
MODEL_INIT_PATH = None  # set to "best.h5" to resume
MODEL_OUT_PATH = "cosinee3.keras"
CSV_LOG_FILE    = "cosinee3.csv" # would be good to make auto index up

sigmoid   = lambda x: 1.0/(1.0+np.exp(-x))
MAX_WH    = 2.0                      # clamp tw/th like your inference script
CONF_THR  = 1.0                      # extra-FP threshold (same as inference)
GRID_SIZE    = 13


# ───────────────── DATA AUGMENTATION ─────────────────

def jitter_image(img: np.ndarray, boxes: List[Tuple[float, float, float, float]],
                 max_jitter: float) -> Tuple[np.ndarray, List[Tuple[float,float,float,float]]]:
    """Random aspect‑ratio + scale jitter"""
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

def color_jitter(img, brightness, contrast, saturation, hue=0.0):
    """
    img: uint8 or float Tensor/ndarray in [0,255] or [0,1]
    brightness, contrast, saturation, hue: floats >= 0 (hue in [0,0.5])
    """
    # 0) bring into float32 [0,1]
    tf_img = tf.image.convert_image_dtype(img, tf.float32)

    # 1) brightness
    if brightness and brightness > 0:
        tf_img = tf.image.random_brightness(tf_img, max_delta=brightness)

    # 2) contrast
    if contrast and contrast > 0:
        lower = 1.0 - contrast
        upper = 1.0 + contrast
        if upper <= lower:
            upper = lower + 1e-6
        tf_img = tf.image.random_contrast(tf_img, lower, upper)

    # 3) saturation
    if saturation and saturation > 0:
        low_sat  = 1.0 - saturation
        high_sat = 1.0 + saturation
        if high_sat <= low_sat:
            high_sat = low_sat + 1e-6
        tf_img = tf.image.random_saturation(tf_img, low_sat, high_sat)

    # 4) hue
    if hue and hue > 0:
        delta = hue if hue > 0 else 1e-6
        tf_img = tf.image.random_hue(tf_img, max_delta=delta)

    # 5) clip to [0,1]
    tf_img = tf.clip_by_value(tf_img, 0.0, 1.0)

    # 6) convert back to uint8 NumPy [0,255] for cv2.resize in letterbox
    tf_img_uint8 = tf.image.convert_image_dtype(tf_img, tf.uint8, saturate=True)
    return tf_img_uint8.numpy()

def letterbox(img: np.ndarray, target: int = IMG_SIZE):
    h, w = img.shape[:2]
    scale = min(target/w, target/h)
    nw, nh = int(w*scale), int(h*scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((target, target, 3), 128, np.uint8)
    pad_x, pad_y = (target - nw)//2, (target - nh)//2
    canvas[pad_y:pad_y+nh, pad_x:pad_x+nw] = resized

    # return pad_x before pad_y
    return canvas.astype(np.float32)/255.0, scale, pad_x, pad_y

# ───────────────── XML → TENSOR ─────────────────

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
    img, scale, pad_x, pad_y = letterbox(img)
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
    """
    Batch-level CSV logger that includes wall-clock timestamp.
    """
    def __init__(self, log_every=50, path=CSV_LOG_FILE):
        super().__init__()
        self.k  = log_every
        self.fp = open(path, "w")
        self.fp.write("epoch,batch,loss,val_loss,timestamp\n")

    def _now(self) -> str:
        return datetime.datetime.now().isoformat(timespec="seconds")

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch + 1

    def on_train_batch_end(self, batch, logs=None):
        if batch % self.k == 0:
            self.fp.write(
                f"{self.epoch},{batch},{logs['loss']:.6f},,"
                f"{self._now()}\n"
            ); self.fp.flush()

    def on_epoch_end(self, epoch, logs=None):
        self.fp.write(
            f"{self.epoch},end,{logs['loss']:.6f},"
            f"{logs.get('val_loss',np.nan):.6f},{self._now()}\n"
        ); self.fp.flush()

    def on_train_end(self, logs=None):
        self.fp.close()

class SaveBestAtEnd(tf.keras.callbacks.Callback):
    def __init__(self, filepath=MODEL_OUT_PATH):
        super().__init__()
        self.best = np.inf
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        val = logs.get("val_loss")
        if val is not None and val < self.best:
            self.best = val
            # immediately save best weights to disk
            self.model.save(self.filepath, include_optimizer=True)
            print(f"\n→ New best model saved (val_loss={val:.4f}) to '{self.filepath}'")

class AugmentScheduler(tf.keras.callbacks.Callback):
    """Rebuilds dataset with new aug params each epoch."""
    def __init__(self, train_paths: List[str], schedule_fn):
        super().__init__()
        self.train_paths = train_paths
        self.schedule_fn = schedule_fn
    def on_epoch_begin(self, epoch, logs=None):
        p = self.schedule_fn(epoch)
        self.model.train_data = build_dataset(self.train_paths, **p)

# ───────────────── COSINE LR + WARM‑UP ─────────────────
class WarmUpCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps, total_steps):
        super().__init__()
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step):
        # ensure float for arithmetic
        step = tf.cast(step, tf.float32)
        warmup = tf.cast(self.warmup_steps, tf.float32)
        total  = tf.cast(self.total_steps,  tf.float32)

        # clamp so we never go past total_steps
        step = tf.minimum(step, total)

        # linear warm-up then cosine decay
        lr = tf.cond(
            step < warmup,
            lambda: self.base_lr * (step / warmup),
            lambda: 0.5 * self.base_lr * (
                1 + tf.cos(
                    np.pi * ((step - warmup) / (total - warmup))
                )
            )
        )
        return lr

    def get_config(self):
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
        }

# ───────────────── MAIN ─────────────────

def main():
    # gather and split XML annotations
    xml_files = sorted(glob.glob(os.path.join(ANNOT_DIR, "*.xml")))
    random.shuffle(xml_files)
    split = int(0.8 * len(xml_files))
    train_xml, val_xml = xml_files[:split], xml_files[split:]

    # # define your per-epoch augmentation schedule
    # def aug_schedule(epoch: int) -> Dict[str, float]:
    #     return {
    #         "max_jitter": 0.1 if epoch < 10 else (0.1 if epoch < 10 else 0.1), # Not sure about
    #         "flip":        epoch >= 0,
    #         "brightness":  0.1 if epoch < 15 else 0.1,
    #         "contrast":    0.1 if epoch < 20 else 0.1,
    #         "saturation":  0.0 if epoch < 25 else 0.0,
    #     }

    # # build the initial datasets (epoch 0 params)
    # train_ds = build_dataset(train_xml, **aug_schedule(0))
    train_ds=build_dataset(train_xml, max_jitter=0.0, flip=False,
                            brightness=0.0, contrast=0.0, saturation=0.0)
    val_ds   = build_dataset(val_xml)

    # instantiate your model
    model = Darknet19(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        num_classes=NUM_CLASSES,
        B=NUM_ANCHORS
    )

    # # configure the Warm-Up + Cosine-Decay learning rate
    steps_per_epoch = len(train_xml) // BATCH_SIZE
    warmup_steps    = steps_per_epoch * 5      # first 5 epochs
    total_steps     = steps_per_epoch * EPOCHS
    base_lr         = 1e-3

    lr_schedule = WarmUpCosineDecay(
        base_lr=base_lr,
        warmup_steps=warmup_steps,
        total_steps=total_steps
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    # optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss=yolo_loss)

    # optionally restore from a checkpoint
    if MODEL_INIT_PATH and os.path.exists(MODEL_INIT_PATH):
        print(f"▶ Restoring weights from {MODEL_INIT_PATH}")
        model.load_weights(MODEL_INIT_PATH)

    callbacks = [
        CSVLogger(log_every=1),
        # AugmentScheduler(train_xml, aug_schedule),
        tf.keras.callbacks.EarlyStopping(
            patience=10,
            monitor="val_loss",
            restore_best_weights=True
        ),
        SaveBestAtEnd()
    ]

    # 8) pass the real training dataset into fit()
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )


if __name__ == "__main__":
    main()