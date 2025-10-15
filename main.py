import argparse
import os
import re
import torch
from torch.utils.data import DataLoader, random_split
from datetime import datetime
from dataset import OCTDataset
from train import train_model
from predict import predict
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from model import get_model
import numpy as np
import random

########################################################
#SEED
########################################################
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False







########################################################
#MAIN
########################################################
def main():
    set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], required=True, help="Mode: 'train' o 'predict'")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")






    ########################################################
    # TRAINING (hiperparámetros)
    ########################################################
    images_dir = 'data/BD_2D/images'
    masks_dir = 'data/BD_2D/mask'
    batch_size = 16
    learning_rate = 1e-4
    num_epochs = 200
    patience = 15





    ########################################################
    # PREDICTION
    ########################################################
    model_filename = '1_train_23-07-25.pth' 
    nombre_volumen_prediccion = "A23654_SCAN2_S1"
    #nombre_volumen_prediccion = "Pilot-Demo_Zirconia_x-mode_images"
    use_test_volume = True







########################################################
#TRANSFORM
########################################################
#CROP
    transform = A.Compose([
        A.Crop(x_min=0, y_min=0, x_max=700, y_max=352),  
        A.PadIfNeeded(min_height=352, min_width=704, border_mode=cv2.BORDER_CONSTANT, value=0),  
        ToTensorV2()  
    ])






########################################################
#DATA
########################################################
    train_images_dir = 'data/BD_2D/train/images'
    train_masks_dir = 'data/BD_2D/train/mask'

    val_images_dir = 'data/BD_2D/val/images'
    val_masks_dir = 'data/BD_2D/val/mask'

   

    model = get_model()




    ########################################################
    #TRAIN
    ########################################################
    if args.mode == 'train':
        train_dataset = OCTDataset(train_images_dir, train_masks_dir, transform=transform)
        val_dataset = OCTDataset(val_images_dir, val_masks_dir, transform=transform)  # O puedes usar otro transform para val (sin aumentos)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        train_output_folder = "trains"
        metrics_output_folder = "metrics"
        os.makedirs(train_output_folder, exist_ok=True)
        os.makedirs(metrics_output_folder, exist_ok=True)

        existing_train_files = os.listdir(train_output_folder)
        existing_train_numbers = [
            int(re.match(r"(\d+)_train_.*\.pth", f).group(1))
            for f in existing_train_files
            if re.match(r"(\d+)_train_.*\.pth", f)
        ]

        next_train_number = max(existing_train_numbers, default=0) + 1

        current_date = datetime.now().strftime("%d-%m-%y")
        model_filename = f'{next_train_number}_train_{current_date}.pth'
        model_save_path = os.path.join(train_output_folder, model_filename)

        best_model_filename = f'{next_train_number}_train_best_{current_date}.pth'
        best_model_save_path = os.path.join(train_output_folder, best_model_filename)

        print(f"Entrenando modelo con nombre: {model_filename}")

        metrics_filename = f'{next_train_number}_train_{current_date}.csv'
        metrics_path = os.path.join(metrics_output_folder, metrics_filename)

        model = train_model(
            model, train_loader, val_loader,
            batch_size, learning_rate, num_epochs,
            device, model_save_path, metrics_path,
            patience=patience,
            run_name=model_filename
        )

        print(f"Métricas guardadas como {metrics_filename}")
        print(f"Último modelo guardado como {model_filename}")
        print(f"Mejor modelo guardado como {best_model_filename}")





    ########################################################
    #PREDICT
    ########################################################
    elif args.mode == 'predict':
        model_filename = '7_train_27-09-25_best.pth'  
        #nombre_volumen_prediccion = "Pilot-Demo_x-mode_images_2"
        nombre_volumen_prediccion = "A23654_SCAN1_S1"

        use_test_volume = True  
        
        numero_entrenamiento = model_filename.split('_')[0]
        if "_best" in model_filename:
            numero_entrenamiento += "_best"

        if use_test_volume:
            predict_images_folder = f"data/BD_2D/val/images/{nombre_volumen_prediccion}"
            predict_masks_folder = f"data/BD_2D/val/mask/{nombre_volumen_prediccion}"
            output_folder = os.path.join("predictions", numero_entrenamiento, nombre_volumen_prediccion)

        else:
            predict_images_folder = f"data/BD_2D/test1/test_images"
            predict_masks_folder = f"data/BD_2D/test1/test_mask"
            output_folder = os.path.join("predictions", numero_entrenamiento)

        os.makedirs(output_folder, exist_ok=True)

        
        match = re.match(r"(\d+)_train_.*\.pth", model_filename)
        if match:
            train_number = int(match.group(1))
        else:
            print("No se pudo extraer el número de entrenamiento del nombre del modelo.")
            exit(1)

        model_save_path = os.path.join("trains", model_filename)
        if not os.path.exists(model_save_path):
            print(f"No se encontró el modelo: {model_save_path}")
            exit(1)

        model.load_state_dict(torch.load(model_save_path, map_location=device))
        print(f"Modelo cargado desde {model_save_path}")
        
        model.to(device)
        print("Guardando predicciones en:", output_folder)

        predict(model, predict_images_folder, predict_masks_folder, output_folder, device, train_number)


if __name__ == "__main__":
    main()

#python main.py --mode predict 
#python main.py --mode train


