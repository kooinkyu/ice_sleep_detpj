import os
import shutil

# 원본 라벨링 데이터 폴더
source_root = "/mnt/d/Users/Brian/Downloads/ice_project/Validation/[라벨]bbox(실제도로환경)/2.승용"

# 복사 대상 폴더
target_dir = "/mnt/d/Users/Brian/Downloads/YOLO_dataset/labels/val"
os.makedirs(target_dir, exist_ok=True)

copy_extensions = [".json", ".JSON", ".txt", ".xml"]
copied = 0

for dirpath, _, filenames in os.walk(source_root):
    for filename in filenames:
        ext = os.path.splitext(filename)[1]
        if ext in copy_extensions:
            src_path = os.path.join(dirpath, filename)
            dst_path = os.path.join(target_dir, filename)

            if os.path.exists(dst_path):
                base, ext2 = os.path.splitext(filename)
                folder_name = os.path.basename(dirpath)
                dst_path = os.path.join(target_dir, f"{folder_name}_{base}{ext2}")
            
            shutil.copy(src_path, dst_path)
            copied += 1
            

print(f"\n✅ 총 {copied}개의 라벨링 파일 복사 완료.")
