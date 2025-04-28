import os
import random
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

# Directories for the Pascal VOC dataset
annotations_dir = "dataset/VOCdevkit/VOC2012/Annotations"
images_dir = "dataset/VOCdevkit/VOC2012/JPEGImages"

# Get a list of all annotation files (XML)
xml_files = [f for f in os.listdir(annotations_dir) if f.endswith(".xml")]

# Randomly choose 3 annotation files
selected_xmls = random.sample(xml_files, 3)
# selected_xmls = ["2007_000032.xml"]
print("Selected XML files:", selected_xmls)

# Process each selected annotation and display the image with bounding boxes
for xml_file in selected_xmls:
    xml_path = os.path.join(annotations_dir, xml_file)
    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get the corresponding image filename from the XML <filename> element.
    # If for some reason this is not available, we fallback to the XML's base name.
    filename_elem = root.find("filename")
    if filename_elem is not None:
        image_filename = filename_elem.text.strip()
    else:
        image_filename = os.path.splitext(xml_file)[0] + ".jpg"
    
    image_path = os.path.join(images_dir, image_filename)
    
    # Load the image using Matplotlib's image reader
    img = mpimg.imread(image_path)
    
    # Create a figure and display the image
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    
    # Loop over each object in the XML annotation and draw its bounding box
    for obj in root.findall("object"):
        # Get the class name for the object
        obj_name = obj.find("name").text.strip()
        
        # Access the bounding box coordinates from <bndbox>
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        box_width = xmax - xmin
        box_height = ymax - ymin
        
        # Create a red rectangle patch for the bounding box
        rect = patches.Rectangle((xmin, ymin), box_width, box_height, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        # Add a text label for the object class near the bounding box
        ax.text(xmin, ymin - 5, obj_name, color='red', fontsize=12, backgroundcolor='white')
    
    # Set the title of the plot and show the image
    plt.title(f"Bounding Boxes for {image_filename}")
    plt.show()