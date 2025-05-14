from PIL import Image
import os

train_img_dir = '/mnt/d/Users/Brian/Downloads/ice_texi_train/images/val'
bad_files = []

for file in os.listdir(train_img_dir):
    if file.endswith('.jpg') or file.endswith('.jpeg'):
        path = os.path.join(train_img_dir, file)
        try:
            with Image.open(path) as img:
                img.verify()
        except Exception as e:
            print(f"Corrupted: {file} -> {e}")
            bad_files.append(file)

print(f"\n총 손상된 파일: {len(bad_files)}개")
