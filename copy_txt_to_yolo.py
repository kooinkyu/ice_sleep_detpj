import json
import os

def convert_bbox(bbox, img_width, img_height):
    x_min, y_min, x_max, y_max = map(float, bbox)
    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    w = (x_max - x_min) / img_width
    h = (y_max - y_min) / img_height
    return x_center, y_center, w, h

def json_to_yolo(json_path, output_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    width = int(data['FileInfo']['Width'])
    height = int(data['FileInfo']['Height'])
    bboxes = data['ObjectInfo']['BoundingBox']

    result_lines = []

    # 왼쪽 눈
    if bboxes['Leye']['isVisible']:
        class_id = 0 if bboxes['Leye']['Opened'] else 1
        coords = convert_bbox(bboxes['Leye']['Position'], width, height)
        result_lines.append(f"{class_id} {' '.join(map(str, coords))}")

    # 오른쪽 눈
    if bboxes['Reye']['isVisible']:
        class_id = 2 if bboxes['Reye']['Opened'] else 3
        coords = convert_bbox(bboxes['Reye']['Position'], width, height)
        result_lines.append(f"{class_id} {' '.join(map(str, coords))}")

    # 입
    if bboxes['Mouth']['isVisible']:
        class_id = 4 if bboxes['Mouth']['Opened'] else 5
        coords = convert_bbox(bboxes['Mouth']['Position'], width, height)
        result_lines.append(f"{class_id} {' '.join(map(str, coords))}")

    # 저장
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    output_file = os.path.join(output_path, base_name + '.txt')
    with open(output_file, 'w') as f:
        f.write('\n'.join(result_lines))

# 전체 폴더 처리
input_dir = '/mnt/d/Users/Brian/Downloads/YOLO_dataset/labels/train'
output_dir = '/mnt/d/Users/Brian/Downloads/YOLO_dataset/labels/train'


os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith('.json'):
        json_path = os.path.join(input_dir, filename)
        json_to_yolo(json_path, output_dir)
