import os
import random
import cv2
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ---------- UTILITY FUNCTIONS ----------

def letterbox_image(image, target_size=(416, 416)):
    """
    Resize the input image to fit within target_size while preserving aspect ratio,
    and add padding (letterboxing) to produce an image of size target_size.
    
    Returns:
        new_image (np.ndarray): The letterboxed (padded) image.
        scale (float): Uniform scale factor used.
        pad_top (int): Vertical padding (pixels) at the top.
        pad_left (int): Horizontal padding (pixels) at the left.
        new_w (int): Resized image width (before padding).
        new_h (int): Resized image height (before padding).
    """
    target_w, target_h = target_size
    orig_h, orig_w, _ = image.shape
    
    # Compute the scale factor (preserving aspect ratio)
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    
    # Resize the image using OpenCV (without padding yet)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create a blank canvas with the target size and a constant gray color (128)
    new_image = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
    
    # Compute padding to center the resized image on the canvas
    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2
    
    # Place the resized image into the canvas (letterbox it)
    new_image[pad_top:pad_top + new_h, pad_left:pad_left + new_w, :] = resized_image
    
    return new_image, scale, pad_top, pad_left, new_w, new_h

def transform_bbox_to_letterbox(bbox, scale, pad_left, pad_top):
    """
    Transform a bounding box from the original image coordinates into the letterboxed image coordinates.
    
    Args:
        bbox (tuple): (xmin, ymin, xmax, ymax) in the original image.
        scale (float): The scaling factor used.
        pad_left (int): The horizontal padding added.
        pad_top (int): The vertical padding added.
    
    Returns:
        tuple: (xmin, ymin, xmax, ymax) in letterboxed image coordinates.
    """
    xmin, ymin, xmax, ymax = bbox
    lb_xmin = xmin * scale + pad_left
    lb_ymin = ymin * scale + pad_top
    lb_xmax = xmax * scale + pad_left
    lb_ymax = ymax * scale + pad_top
    return lb_xmin, lb_ymin, lb_xmax, lb_ymax

def inverse_transform_bbox(lb_bbox, scale, pad_left, pad_top):
    """
    Inverse transform a bounding box from letterboxed image coordinates back to the original image coordinates.
    
    Args:
        lb_bbox (tuple): (xmin, ymin, xmax, ymax) in letterboxed coordinates.
        scale (float): The scaling factor used.
        pad_left (int): Horizontal padding added.
        pad_top (int): Vertical padding added.
        
    Returns:
        tuple: (xmin, ymin, xmax, ymax) in the original image coordinates.
    """
    lb_xmin, lb_ymin, lb_xmax, lb_ymax = lb_bbox
    orig_xmin = (lb_xmin - pad_left) / scale
    orig_ymin = (lb_ymin - pad_top) / scale
    orig_xmax = (lb_xmax - pad_left) / scale
    orig_ymax = (lb_ymax - pad_top) / scale
    return orig_xmin, orig_ymin, orig_xmax, orig_ymax

def draw_bboxes(ax, bboxes, color="red", lw=2, labels=None):
    """
    Draw bounding boxes on a matplotlib axis.
    
    Args:
        ax (matplotlib.axes.Axes): The axis on which to draw.
        bboxes (list of tuples): Each tuple is (xmin, ymin, xmax, ymax).
        color (str): Color of the bounding box.
        lw (int): Line width.
        labels (list of str): Optional list of labels for each bounding box.
    """
    for i, bbox in enumerate(bboxes):
        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=lw, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        if labels is not None and i < len(labels):
            ax.text(xmin, max(ymin - 5, 0), labels[i], color=color, fontsize=12, backgroundcolor='white')

def parse_voc_annotation(xml_file):
    """
    Parse a VOC XML file to extract the image filename and bounding boxes.
    
    Args:
        xml_file (str): Path to the XML file.
        
    Returns:
        tuple: (image_filename, bboxes, labels)
            image_filename (str): The image filename (as given in the XML).
            bboxes (list of tuples): Each is (xmin, ymin, xmax, ymax) as integers.
            labels (list of str): The object class names.
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    filename = root.find("filename").text.strip()
    bboxes = []
    labels = []
    for obj in root.findall("object"):
        label = obj.find("name").text.strip()
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        bboxes.append((xmin, ymin, xmax, ymax))
        labels.append(label)
    return filename, bboxes, labels

# ---------- MAIN TEST SUITE ----------

# Directories for JPEG images and XML annotations.
images_dir = "dataset/VOCdevkit/VOC2012/JPEGImages"
ann_dir    = "dataset/VOCdevkit/VOC2012/Annotations"

# Gather all XML files (assume each image we select has an annotation).
xml_files = [os.path.join(ann_dir, f) for f in os.listdir(ann_dir) if f.lower().endswith(".xml")]

# Randomly select 3 XML files.
selected_xmls = random.sample(xml_files, 3)

# Prepare lists to hold per-image data.
orig_images = []
letterboxed_images = []
bboxes_orig_list = []
bboxes_lb_list = []
bboxes_recovered_list = []
labels_list = []
info_list = []  # strings to display info

# Process each selected XML file.
for xml_path in selected_xmls:
    try:
        img_filename, bboxes_orig, labels = parse_voc_annotation(xml_path)
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        continue
    
    # Build image path and load the image.
    img_path = os.path.join(images_dir, img_filename)
    orig_img = cv2.imread(img_path)
    if orig_img is None:
        print(f"Warning: Could not load image {img_filename}")
        continue
    # Convert from BGR to RGB for display.
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    
    # Apply letterbox transformation.
    target_size = (416, 416)
    lb_img, scale, pad_top, pad_left, new_w, new_h = letterbox_image(orig_img, target_size)
    
    # Transform original bounding boxes to letterbox image coordinates.
    bboxes_lb = [transform_bbox_to_letterbox(bb, scale, pad_left, pad_top) for bb in bboxes_orig]
    
    # Inverse transform: Recover bounding boxes back to original image coordinates.
    bboxes_recovered = [inverse_transform_bbox(bb, scale, pad_left, pad_top) for bb in bboxes_lb]
    
    # Save data.
    orig_images.append(orig_img)
    letterboxed_images.append(lb_img)
    bboxes_orig_list.append(bboxes_orig)
    bboxes_lb_list.append(bboxes_lb)
    bboxes_recovered_list.append(bboxes_recovered)
    labels_list.append(labels)
    
    # Create an information string for this image.
    info_str = (
        f"Original dims: {orig_img.shape[1]}x{orig_img.shape[0]}\n"
        f"Resized dims (before pad): {new_w}x{new_h}\n"
        f"Target dims: 416x416\n"
        f"Scale: {scale:.2f}\n"
        f"Padding (top,left): ({pad_top}, {pad_left})"
    )
    info_list.append(info_str)

# Set up a figure with 3 rows and 3 columns.
num_images = len(orig_images)
fig, axes = plt.subplots(num_images, 3, figsize=(18, 6 * num_images))

if num_images == 1:
    axes = np.expand_dims(axes, axis=0)

# For each image, plot the 3 panels.
for i in range(num_images):
    # Panel 1: Original image with original bounding boxes.
    axes[i, 0].imshow(orig_images[i])
    axes[i, 0].set_title(f"Original Image\n{info_list[i]}")
    draw_bboxes(axes[i, 0], bboxes_orig_list[i], color="red", lw=2, labels=labels_list[i])
    axes[i, 0].axis("off")
    
    # Panel 2: Letterboxed image with bounding boxes (transformed).
    axes[i, 1].imshow(letterboxed_images[i])
    axes[i, 1].set_title("Letterboxed Image (416x416)")
    draw_bboxes(axes[i, 1], bboxes_lb_list[i], color="blue", lw=2, labels=labels_list[i])
    axes[i, 1].axis("off")
    
    # Panel 3: Original image with recovered bounding boxes.
    axes[i, 2].imshow(orig_images[i])
    axes[i, 2].set_title("Recovered Bounding Boxes (Inverse Transform)")
    draw_bboxes(axes[i, 2], bboxes_recovered_list[i], color="green", lw=2, labels=labels_list[i])
    axes[i, 2].axis("off")

plt.tight_layout()
plt.show()