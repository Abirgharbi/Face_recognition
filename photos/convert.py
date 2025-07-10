from PIL import Image
import os

def convert_images_to_rgb(folder):
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder, filename)
            try:
                img = Image.open(img_path).convert('RGB')
                img.save(img_path)  # overwrite with RGB version
                print(f"✅ Converted {filename} to RGB")
            except Exception as e:
                print(f"❌ Failed to convert {filename}: {e}")

convert_images_to_rgb('.')

