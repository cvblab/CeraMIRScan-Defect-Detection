import os
import cv2
from torch.utils.data import Dataset
import numpy as np

class OCTDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.samples = []

        valid_exts = {".png", ".jpg", ".jpeg", ".tif"}  

        for scan_folder in os.listdir(images_dir):
            img_folder_path = os.path.join(images_dir, scan_folder)
            mask_folder_path = os.path.join(masks_dir, scan_folder)

            if os.path.isdir(img_folder_path):
                for filename in os.listdir(img_folder_path):
                    ext = os.path.splitext(filename)[1].lower()
                    if ext not in valid_exts:
                        continue  

                    img_path = os.path.join(img_folder_path, filename)
                    mask_path = os.path.join(mask_folder_path, filename)

                    if os.path.exists(mask_path):
                        self.samples.append((img_path, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {img_path}")
        if mask is None:
            raise FileNotFoundError(f"No se pudo leer la máscara: {mask_path}")

        image = image.astype("float32") / 255.0

        mask = (mask > 0).astype("float32")

        height, width = image.shape
        x_max_crop = min(700, width)
        y_max_crop = min(352, height)

        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        transform = A.Compose([
            A.Crop(x_min=0, y_min=0, x_max=x_max_crop, y_max=y_max_crop),
            A.PadIfNeeded(min_height=352, min_width=704, border_mode=cv2.BORDER_CONSTANT, value=0),
            ToTensorV2()
        ])

        augmented = transform(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]

        return image, mask.unsqueeze(0)
