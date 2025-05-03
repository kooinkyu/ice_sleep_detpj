import os
import shutil

source_root = "/mnt/d/Users/Brian/Downloads/ice_project/Validation/[원천]bbox(실제도로환경)/2.승용"
target_dir = "/mnt/d/Users/Brian/Downloads/YOLO_dataset/images/val"
os.makedirs(target_dir, exist_ok=True)

for folder in os.listdir(source_root):
    folder_path = os.path.join(source_root, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            ext = os.path.splitext(file)[1].lower().strip()
            if ext == ".jpg":
                image_path = os.path.join(folder_path, file)
                filename = os.path.basename(image_path)
                destination_path = os.path.join(target_dir, filename)

                if os.path.exists(destination_path):
                    base, ext = os.path.splitext(filename)
                    destination_path = os.path.join(target_dir, f"{folder}_{base}{ext}")

                
                shutil.copy(image_path, destination_path)

print("✅ 모든 이미지가 성공적으로 복사되었습니다.")
