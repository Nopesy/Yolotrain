import os, random
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow as tf

from Code0.yolo import Darknet19, yolo_loss  # your definitions

# Paths – adjust as needed
MODEL_PATH = "best_yolo.h5"
IMAGE_DIR  = "dataset/VOCdevkit/VOC2012/JPEGImages"
CLASS_NAMES = ["aeroplane","bicycle","bird","boat","bottle",
               "bus","car","cat","chair","cow","diningtable",
               "dog","horse","motorbike","person","pottedplant",
               "sheep","sofa","train","tvmonitor"]
GRID_SIZE = 13
INPUT_SIZE = 416
OBJ_THRESHOLD = 0.05

# 1) Load model
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={'yolo_loss': yolo_loss}
)

# 2) Letterbox util
def letterbox(img, target_size=416):
    h,w = img.shape[:2]
    scale = min(target_size/w, target_size/h)
    nw,nh = int(w*scale), int(h*scale)
    resized = cv2.resize(img, (nw,nh))
    canvas = np.full((target_size,target_size,3),128, dtype=np.uint8)
    dx,dy = (target_size-nw)//2, (target_size-nh)//2
    canvas[dy:dy+nh,dx:dx+nw] = resized
    return canvas, scale, dx, dy

# 3) Inference + decode
def decode_and_draw(img_path):
    # Load + BGR→RGB
    orig = cv2.imread(img_path)
    img = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    # Letterbox
    lb, scale, dx, dy = letterbox(img, INPUT_SIZE)
    inp = lb.astype(np.float32)/255.0
    # Predict
    pred = model.predict(inp[None,...])[0]  # shape (13,13,1,5+20)
    fig, ax = plt.subplots(1,1,figsize=(6,6))
    ax.imshow(lb)
    cell_w = cell_h = INPUT_SIZE / GRID_SIZE

    # Loop over grid cells
    for j in range(GRID_SIZE):
        for i in range(GRID_SIZE):
            tx, ty, tw, th, to = pred[j,i,0, :5]
            if to < OBJ_THRESHOLD:
                continue
            # recover center in pixels within letterbox
            cx = (i + tx) * cell_w
            cy = (j + ty) * cell_h
            bw = tw * cell_w
            bh = th * cell_h
            # corners
            x1 = cx - bw/2
            y1 = cy - bh/2
            # class
            class_probs = pred[j,i,0, 5:]
            cls = np.argmax(class_probs)
            name = CLASS_NAMES[cls]
            # draw
            rect = patches.Rectangle(
                (x1,y1), bw, bh,
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(
                x1, y1-6, f"{name}:{to:.2f}",
                color='white', fontsize=8, backgroundcolor='red'
            )

    ax.axis('off')
    plt.show()

# 4) Run on 3 random images
imgs = [os.path.join(IMAGE_DIR,f) 
        for f in os.listdir(IMAGE_DIR) 
        if f.endswith('.jpg')]
for img_path in random.sample(imgs, 3):
    decode_and_draw(img_path)