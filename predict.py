
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import pandas as pd
from skimage.measure import label


def calculate_metrics(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    if gt.sum() == 0 and pred.sum() == 0:
        return 1.0, 1.0, 1.0, 1.0  # precision, recall, dice, iou

    tp = np.logical_and(pred, gt).sum()
    fp = np.logical_and(pred, ~gt).sum()
    fn = np.logical_and(~pred, gt).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return precision, recall, dice, iou


def contar_falsos_positivos(gt_mask, pred_mask, min_overlap=1):
    pred_labeled = label(pred_mask.astype(bool))
    total_pred = pred_labeled.max()
    falsos_positivos = 0

    for region_id in range(1, total_pred + 1):
        region = (pred_labeled == region_id)
        if np.logical_and(region, gt_mask.astype(bool)).sum() < min_overlap:
            falsos_positivos += 1

    return total_pred, falsos_positivos


def contar_errores_detectados(gt_mask, pred_mask, min_overlap=1):
    gt_labeled = label(gt_mask.astype(bool))
    total_errores = gt_labeled.max()
    detectados = 0
    detectados_ids = set()

    for region_id in range(1, total_errores + 1):
        region = (gt_labeled == region_id)
        if np.logical_and(region, pred_mask.astype(bool)).sum() >= min_overlap:
            detectados += 1
            detectados_ids.add(region_id)

    return total_errores, detectados, detectados_ids


def predict(model, predict_images_folder, predict_masks_folder, output_folder, device, train_number=None):
    model.eval()
    os.makedirs(output_folder, exist_ok=True)

    image_files = sorted(os.listdir(predict_images_folder))
    print(f"Se encontraron {len(image_files)} imágenes en {predict_images_folder}")

    results = []

    with torch.no_grad():
        for img_file in tqdm(image_files, desc="Prediciendo"):
            img_path = os.path.join(predict_images_folder, img_file)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"No se pudo cargar la imagen: {img_path}")
                continue

            mask_path = os.path.join(predict_masks_folder, img_file)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"No se pudo cargar la máscara: {mask_path}")
                continue

            start_time = time.time()

            image = image[0:352, 0:700]
            image = cv2.copyMakeBorder(image, 0, 0, 0, 4, borderType=cv2.BORDER_CONSTANT, value=0)
            mask = mask[0:352, 0:700]
            mask = cv2.copyMakeBorder(mask, 0, 0, 0, 4, borderType=cv2.BORDER_CONSTANT, value=0)

            image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0
            output = model(image_tensor)
            prob_map = torch.sigmoid(output).cpu().squeeze().numpy()
            binary_pred = (prob_map > 0.5).astype(np.uint8) * 255

            end_time = time.time()
            inference_time = end_time - start_time

            mask_bin = (mask > 127).astype(np.uint8)
            pred_bin = (binary_pred > 127).astype(np.uint8)
            precision, recall, dice, iou = calculate_metrics(mask_bin, pred_bin)

            total_err, err_detect, ids_detectados = contar_errores_detectados(mask_bin, pred_bin)
            errores_no_detectados = total_err - err_detect

            _, falsos_positivos = contar_falsos_positivos(mask_bin, pred_bin)


            results.append({
                "Imagen": img_file,
                "Precision": precision,
                "Recall": recall,
                "Dice": dice,
                "IoU": iou,
                "TotalErroresGT": total_err,
                "ErroresDetectados": err_detect,
                "ErroresNoDetectados": errores_no_detectados,
                "FalsosPositivos": falsos_positivos,
                "InferenceTime": inference_time
            })


            print(f"{img_file} - Prec: {precision:.4f} | Rec: {recall:.4f} | Dice: {dice:.4f} | IoU: {iou:.4f}")

            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            red_layer = np.zeros_like(image_rgb, dtype=np.uint8)
            pred_bool = binary_pred > 0
            red_layer[pred_bool] = [255, 0, 0]
            overlay = cv2.addWeighted(image_rgb, 1.0, red_layer, 0.5, 0)

            gt_bool = mask > 0
            gt_visible = np.logical_and(gt_bool, ~pred_bool)
            overlay[gt_visible] = [255, 255, 255]

            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            axs = axs.flatten()
            axs[0].imshow(image, cmap='gray'); axs[0].set_title("Imagen original"); axs[0].axis('off')
            axs[1].imshow(mask, cmap='gray'); axs[1].set_title("Máscara GT"); axs[1].axis('off')
            axs[2].imshow(binary_pred, cmap='gray'); axs[2].set_title("Predicción"); axs[2].axis('off')
            axs[3].imshow(overlay); axs[3].set_title("Superposición"); axs[3].axis('off')
            plt.figtext(0.5, 0.01, f"Inferencia: {inference_time:.4f} s", ha="center", fontsize=10)

            pred_image_name = f"{train_number}_composite_{os.path.splitext(img_file)[0]}.png"
            save_path = os.path.join(output_folder, pred_image_name)
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()

            print(f"Guardado: {save_path}")
            print(f"Tiempo de inferencia: {inference_time:.4f} segundos")

    metrics_csv = os.path.join(output_folder, f"{train_number}_metrics.csv")
    df = pd.DataFrame(results)

    metrics_mean = df[["Precision", "Recall", "Dice", "IoU", "InferenceTime"]].mean()

    summary_row = pd.DataFrame({
        "Imagen": ["Mean"],
        "Precision": [metrics_mean["Precision"]],
        "Recall": [metrics_mean["Recall"]],
        "Dice": [metrics_mean["Dice"]],
        "IoU": [metrics_mean["IoU"]],
        "TotalErroresGT": [df["TotalErroresGT"].sum()],
        "ErroresDetectados": [df["ErroresDetectados"].sum()],
        "ErroresNoDetectados": [df["ErroresNoDetectados"].sum()],
        "FalsosPositivos": [df["FalsosPositivos"].sum()],
        "InferenceTime": [metrics_mean["InferenceTime"]]
    })

    
    df_final = pd.concat([df, summary_row], ignore_index=True)

    df_final.to_csv(metrics_csv, index=False)
    print(f"Métricas guardadas en: {metrics_csv}")


