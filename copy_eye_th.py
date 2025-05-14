import json
import os

def convert_bbox(bbox, img_width, img_height):
    x_min, y_min, x_max, y_max = map(float, bbox)
    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    w = (x_max - x_min) / img_width
    h = (y_max - y_min) / img_height
    return x_center, y_center, w, h

def merge_bboxes(b1, b2):
    x_min = min(float(b1[0]), float(b2[0]))
    y_min = min(float(b1[1]), float(b2[1]))
    x_max = max(float(b1[2]), float(b2[2]))
    y_max = max(float(b1[3]), float(b2[3]))
    return [x_min, y_min, x_max, y_max]

def json_to_yolo(json_path, output_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    width = int(data['FileInfo']['Width'])
    height = int(data['FileInfo']['Height'])
    bboxes = data['ObjectInfo']['BoundingBox']

    result_lines = []

    # 👁️ 눈 처리 (왼쪽과 오른쪽 눈 둘 다 존재할 때만)
    leye = bboxes.get('Leye')
    reye = bboxes.get('Reye')
    if leye['isVisible'] and reye['isVisible']:
        is_open = leye['Opened'] or reye['Opened']
        class_id = 0 if is_open else 1  # 0: 눈 열림, 1: 눈 닫힘
        merged_box = merge_bboxes(leye['Position'], reye['Position'])
        coords = convert_bbox(merged_box, width, height)
        result_lines.append(f"{class_id} {' '.join(map(str, coords))}")

    # 👄 입 처리 (그대로)
    mouth = bboxes.get('Mouth')
    if mouth['isVisible']:
        class_id = 2 if mouth['Opened'] else 3  # 2: 입 열림, 3: 입 닫힘
        coords = convert_bbox(mouth['Position'], width, height)
        result_lines.append(f"{class_id} {' '.join(map(str, coords))}")

    # 저장
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    output_file = os.path.join(output_path, base_name + '.txt')
    with open(output_file, 'w') as f:
        f.write('\n'.join(result_lines))

# 전체 폴더 처리
input_dir = '/mnt/d/Users/Brian/Downloads/ice_texi_train/labels/val'
output_dir = '/mnt/d/Users/Brian/Downloads/ice_texi_train/labels/val_txt'

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith('.json'):
        json_path = os.path.join(input_dir, filename)
        json_to_yolo(json_path, output_dir)
