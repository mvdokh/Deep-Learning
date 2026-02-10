import os
import json
from tqdm import tqdm

# Set your paths here
ANNOTATIONS_DIR = '/home/wanglab/PyTorch/Keypoints/glue_tubes_keypoints_dataset_134imgs/train/annotations'
IMAGES_DIR = '/home/wanglab/PyTorch/Keypoints/glue_tubes_keypoints_dataset_134imgs/train/images'
COCO_OUT_DIR = '/home/wanglab/PyTorch/Keypoints/glue_tubes_keypoints_dataset_134imgs/train_coco'
COCO_OUT_PATH = os.path.join(COCO_OUT_DIR, 'annotations.json')

os.makedirs(COCO_OUT_DIR, exist_ok=True)

# COCO template
coco = {
    "images": [],
    "annotations": [],
    "categories": [
        {"id": 1, "name": "Glue tube"}
    ]
}

ann_id = 1
image_id_map = {}

annotation_files = sorted(os.listdir(ANNOTATIONS_DIR))
for idx, ann_file in enumerate(tqdm(annotation_files)):
    if not ann_file.endswith('.json'):
        continue
    ann_path = os.path.join(ANNOTATIONS_DIR, ann_file)
    with open(ann_path) as f:
        ann = json.load(f)
    # Assume image filename is the same as the annotation file but with .jpg or .png
    base_name = os.path.splitext(ann_file)[0]
    for ext in ['.jpg', '.jpeg', '.png']:
        img_path = os.path.join(IMAGES_DIR, base_name + ext)
        if os.path.exists(img_path):
            file_name = base_name + ext
            break
    else:
        print(f"Image for {ann_file} not found!")
        continue
    image_id = idx + 1
    image_id_map[base_name] = image_id
    # You may want to get image size using cv2 or PIL
    try:
        import cv2
        img = cv2.imread(img_path)
        height, width = img.shape[:2]
    except:
        height, width = 0, 0
    coco["images"].append({
        "id": image_id,
        "file_name": file_name,
        "height": height,
        "width": width
    })
    # Each bbox in ann['bboxes']
    for bbox in ann.get('bboxes', []):
        # COCO expects [x, y, width, height]
        x, y, x2, y2 = bbox
        coco_bbox = [x, y, x2 - x, y2 - y]
        coco["annotations"].append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": 1,
            "bbox": coco_bbox,
            "area": (x2 - x) * (y2 - y),
            "iscrowd": 0
        })
        ann_id += 1

with open(COCO_OUT_PATH, 'w') as f:
    json.dump(coco, f, indent=2)

print(f"COCO annotation file saved to {COCO_OUT_PATH}")
