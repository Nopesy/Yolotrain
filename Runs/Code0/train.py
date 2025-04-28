import os
import glob
import random
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
import cv2
from Code0.yolo import Darknet19, yolo_loss

# ——— Configuration ———
ANNOT_DIR    = "dataset/VOCdevkit/VOC2012/Annotations"
IMAGE_DIR    = "dataset/VOCdevkit/VOC2012/JPEGImages"
CLASS_NAMES  = ["aeroplane","bicycle","bird","boat","bottle",
                "bus","car","cat","chair","cow","diningtable",
                "dog","horse","motorbike","person","pottedplant",
                "sheep","sofa","train","tvmonitor"]
NUM_CLASSES  = len(CLASS_NAMES)
GRID_SIZE    = 13
INPUT_SIZE   = 416
BATCH_SIZE   = 8
EPOCHS       = 5

# Optional: initialize from existing .h5
MODEL_INIT_PATH = "yolov2_singlebox.h5"  # set to None to train from scratch

# ——— Data Augmentation: Aspect‑ratio jitter ———
def random_aspect_jitter(image, bboxes, max_jitter=0.1):
    h, w = image.shape[:2]
    jx = random.uniform(1 - max_jitter, 1 + max_jitter)
    jy = random.uniform(1 - max_jitter, 1 + max_jitter)
    new_w = int(w * jx)
    new_h = int(h * jy)
    image = cv2.resize(image, (new_w, new_h))
    bboxes = [(xmin * jx,
               ymin * jy,
               xmax * jx,
               ymax * jy)
              for (xmin, ymin, xmax, ymax) in bboxes]
    return image, bboxes

# ——— Letter‑boxing ———
def letterbox_and_preprocess(img, target_size=(INPUT_SIZE,INPUT_SIZE)):
    h, w = img.shape[:2]
    scale = min(target_size[0]/w, target_size[1]/h)
    nw, nh = int(w*scale), int(h*scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((target_size[1], target_size[0], 3), 128, dtype=np.uint8)
    pad_x, pad_y = (target_size[0]-nw)//2, (target_size[1]-nh)//2
    canvas[pad_y:pad_y+nh, pad_x:pad_x+nw] = resized
    return canvas.astype(np.float32)/255.0, scale, pad_y, pad_x

# ——— Parsing + Encoding ———
def parse_example(xml_path_str):
    # xml_path_str is a Tensor; convert to Python bytes then string
    xml_bytes = xml_path_str.numpy() if hasattr(xml_path_str, 'numpy') else xml_path_str
    xml_path = xml_bytes.decode('utf-8') if isinstance(xml_bytes, (bytes, bytearray)) else xml_bytes
    tree = ET.parse(xml_path)
    root = tree.getroot()
    fname = root.find('filename').text
    raw_bboxes, raw_labels = [], []
    for obj in root.findall('object'):
        raw_labels.append(obj.find('name').text)
        b = obj.find('bndbox')
        raw_bboxes.append((
            float(b.find('xmin').text),
            float(b.find('ymin').text),
            float(b.find('xmax').text),
            float(b.find('ymax').text)
        ))
    # load image
    img = cv2.imread(os.path.join(IMAGE_DIR, fname))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # augment + preprocess
    img, raw_bboxes = random_aspect_jitter(img, raw_bboxes)
    inp, scale, pad_y, pad_x = letterbox_and_preprocess(img)
    # build y_true
    y_true = np.zeros((GRID_SIZE, GRID_SIZE, 1, 5+NUM_CLASSES), dtype=np.float32)
    for (xmin, ymin, xmax, ymax), cls in zip(raw_bboxes, raw_labels):
        cid = CLASS_NAMES.index(cls)
        xmin = xmin*scale + pad_x
        ymin = ymin*scale + pad_y
        xmax = xmax*scale + pad_x
        ymax = ymax*scale + pad_y
        bw, bh = xmax-xmin, ymax-ymin
        x_ctr, y_ctr = xmin + bw/2, ymin + bh/2
        x_n, y_n = x_ctr/INPUT_SIZE, y_ctr/INPUT_SIZE
        col = int(x_n * GRID_SIZE)
        row = int(y_n * GRID_SIZE)
        off_x = x_n * GRID_SIZE - col
        off_y = y_n * GRID_SIZE - row
        w_cells = bw / (INPUT_SIZE/GRID_SIZE)
        h_cells = bh / (INPUT_SIZE/GRID_SIZE)
        y_true[row, col, 0, 0:2] = [off_x, off_y]
        y_true[row, col, 0, 2:4] = [w_cells, h_cells]
        y_true[row, col, 0, 4] = 1.0
        y_true[row, col, 0, 5+cid] = 1.0
    return inp, y_true

# ——— tf.data pipeline using parallel map ———
all_xmls = sorted(glob.glob(os.path.join(ANNOT_DIR, "*.xml")))
random.shuffle(all_xmls)
split_idx = int(0.8 * len(all_xmls))
train_paths = all_xmls[:split_idx]
val_paths   = all_xmls[split_idx:]

@tf.function
def load_and_preprocess(xml_path):
    inp, y = tf.py_function(
        func=lambda p: parse_example(p),
        inp=[xml_path],
        Tout=(tf.float32, tf.float32)
    )
    inp.set_shape((INPUT_SIZE, INPUT_SIZE, 3))
    y.set_shape((GRID_SIZE, GRID_SIZE, 1, 5+NUM_CLASSES))
    return inp, y

train_ds = tf.data.Dataset.from_tensor_slices(train_paths)
train_ds = (
    train_ds.shuffle(len(train_paths))
            .map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
)
val_ds = tf.data.Dataset.from_tensor_slices(val_paths)
val_ds = (
    val_ds.shuffle(len(val_paths))
          .map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
          .batch(BATCH_SIZE, drop_remainder=True)
          .prefetch(tf.data.AUTOTUNE)
)

model = Darknet19(input_shape=(INPUT_SIZE,INPUT_SIZE,3),
                  num_classes=NUM_CLASSES, B=1)
if MODEL_INIT_PATH and os.path.exists(MODEL_INIT_PATH):
    print(f"Loading weights from {MODEL_INIT_PATH}")
    model = tf.keras.models.load_model(
        MODEL_INIT_PATH,
        custom_objects={'yolo_loss': yolo_loss}
    )
else:
    model.compile(optimizer='adam', loss=yolo_loss)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint('best_yolo.h5', monitor='val_loss', save_best_only=True, verbose=1),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
]

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
model.save('yolov2_singlebox_final.h5')
