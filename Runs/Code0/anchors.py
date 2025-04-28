# generate_anchors.py

import os
import glob
import xml.etree.ElementTree as ET
import numpy as np
from sklearn.cluster import KMeans

# ——— CONFIG ———
ANNOT_DIR   = "dataset/VOCdevkit/VOC2012/Annotations"
INPUT_SIZE  = (416, 416)   # width, height
NUM_CLUSTERS = 3           # e.g. 9 anchors (you can choose 6, 9, etc.)

def load_boxes(annotation_dir):
    wh = []
    for xml_file in glob.glob(os.path.join(annotation_dir, "*.xml")):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        W = float(root.find("size/width").text)
        H = float(root.find("size/height").text)
        for obj in root.findall("object"):
            b = obj.find("bndbox")
            xmin = float(b.find("xmin").text)
            ymin = float(b.find("ymin").text)
            xmax = float(b.find("xmax").text)
            ymax = float(b.find("ymax").text)
            bw = (xmax - xmin) / W
            bh = (ymax - ymin) / H
            wh.append([bw, bh])
    return np.array(wh)

def cluster_anchors(box_wh, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(box_wh)
    anchors = kmeans.cluster_centers_
    # convert to absolute pixel sizes on INPUT_SIZE
    abs_anchors = anchors * np.array(INPUT_SIZE)[None, :]
    # sort by area
    areas = abs_anchors[:, 0] * abs_anchors[:, 1]
    order = np.argsort(areas)
    return abs_anchors[order]

if __name__ == "__main__":
    boxes = load_boxes(ANNOT_DIR)
    anchors = cluster_anchors(boxes, NUM_CLUSTERS)
    print("Anchors (w, h) in pixels:")
    for w, h in anchors:
        print(f"{w:.1f}, {h:.1f}")