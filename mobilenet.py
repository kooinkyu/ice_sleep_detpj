import os
import json
import cv2
from pathlib import Path

# === 경로 설정 ===
base_dir   = "/home/brian/Downloads/ice_mobilnet_data/Validation"
label_root = os.path.join(base_dir, "라벨_bbox(실제도로환경)")
image_root = os.path.join(base_dir, "원천_bbox(실제도로환경)")
output_dir = os.path.join(base_dir, "cropped_real")

# ✅ 승용만 처리
TARGET_FOLDER_NAME = "2.승용"

os.makedirs(output_dir, exist_ok=True)

# === 부위명, 클래스 매핑 ===
bbox_targets = {
    "Leye": "eye",
    "Reye": "eye",
    "Mouth": "mouth",
}
def crop_from_json(image_path, json_path, save_dir):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ 이미지 불러오기 실패: {image_path}")
        return

    H, W = image.shape[:2]

    def to_int(x):
        # '288.00' 같은 문자열/float 모두 처리
        try:
            return int(round(float(x)))
        except Exception:
            return None

    bbox = data["ObjectInfo"]["BoundingBox"]
    for part, label_prefix in bbox_targets.items():
        info = bbox.get(part)
        if not info:
            continue

        # isVisible이 문자열일 수도 있으니 안전하게 처리
        is_visible = info.get("isVisible", True)
        if isinstance(is_visible, str):
            is_visible = is_visible.lower() == "true"
        if not is_visible:
            continue

        pos = info.get("Position")
        if not pos or len(pos) != 4:
            continue

        x1, y1, x2, y2 = [to_int(v) for v in pos]
        if None in (x1, y1, x2, y2):
            continue

        # 혹시 순서가 뒤집혀 있으면 교정
        if x2 < x1: x1, x2 = x2, x1
        if y2 < y1: y1, y2 = y2, y1

        # 이미지 경계로 클립
        x1 = max(0, min(x1, W - 1))
        x2 = max(0, min(x2, W))
        y1 = max(0, min(y1, H - 1))
        y2 = max(0, min(y2, H))

        # 유효성 체크(너무 작은 bbox 제외)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue

        opened = info.get("Opened", False)
        if isinstance(opened, str):
            opened = opened.lower() == "true"

        label = f"{label_prefix}_{'open' if opened else 'closed'}"
        save_subdir = os.path.join(save_dir, label)
        os.makedirs(save_subdir, exist_ok=True)

        crop = image[y1:y2, x1:x2]
        img_name = Path(image_path).stem + f"_{part}.jpg"
        save_path = os.path.join(save_subdir, img_name)
        cv2.imwrite(save_path, crop)


# === 전체 폴더 순회 (재귀) ===
# label_root 아래에서 오직 '2.승용' 트리만 돈다
for root, dirs, files in os.walk(label_root):
    rel = os.path.relpath(root, label_root)  # label_root 기준 상대 경로
    top = rel.split(os.sep)[0] if rel != "." else ""

    # 최상위 카테고리가 '2.승용'이 아니면 스킵
    if top and top != TARGET_FOLDER_NAME:
        continue

    # 현재 라벨 폴더에 대응되는 원천 이미지 폴더 경로
    image_folder = os.path.join(image_root, rel if rel != "." else "")
    if not os.path.isdir(image_folder):
        # 대응 이미지 폴더가 없으면 스킵
        continue

    for file in files:
        if not file.endswith(".json"):
            continue

        json_path = os.path.join(root, file)

        # 파일명 기준 .jpg → .JPG 대체 확인
        base_name = file[:-5]  # strip ".json"
        image_path = os.path.join(image_folder, base_name + ".jpg")
        if not os.path.exists(image_path):
            image_path_alt = os.path.join(image_folder, base_name + ".JPG")
            if os.path.exists(image_path_alt):
                image_path = image_path_alt
            else:
                print(f"[경고] 이미지 없음: {image_path} 또는 {image_path_alt}")
                continue

        crop_from_json(image_path, json_path, output_dir)

print("✅ 승용(2.승용)만 크롭 완료!")
