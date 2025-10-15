import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import time
import csv
import os
from datetime import datetime
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
from segmentation_models_pytorch.utils.metrics import Fscore, Precision, Recall
import wandb

def train_model(model, train_loader, val_loader, batch_size, learning_rate, num_epochs, device, model_path, metrics_path, patience, run_name=None):
    import wandb
    wandb.init(
        project="final_model",
        name=run_name,  
        config={
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": num_epochs,
            "model": model.__class__.__name__,
        }
    )


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


#############  FUNCIÓN DE PÉRDIDA.  ##################
    bce_loss = smp.losses.SoftBCEWithLogitsLoss()
    dice_loss = smp.losses.DiceLoss(mode='binary')
    def loss_combined(pred, target):
        return 0.5 * dice_loss(pred, target) + 0.5 * bce_loss(pred, target)
        #return 0.3 * dice_loss(pred, target) + 0.7 * bce_loss(pred, target)
        #return 0.7 * dice_loss(pred, target) + 0.3 * bce_loss(pred, target)
        #return bce_loss(pred, target)
        #return dice_loss(pred, target)

    loss_fn = loss_combined
#######################################################




    model.to(device)

    dice_metric = Fscore(threshold=0.5)
    precision_metric = Precision(threshold=0.5)
    recall_metric = Recall(threshold=0.5)

    os.makedirs("metrics/grafics", exist_ok=True)

    with open(metrics_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Epoch", "Train Loss", "Validation Loss", "Dice", "Precision", "Recall", "Time (s)", "Batch Size", "Learning Rate"])

    train_losses, val_losses = [], []
    dice_scores, precision_scores, recall_scores = [], [], []

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_model_path = model_path.replace(".pth", "_best.pth")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(device, dtype=torch.float)
            masks = masks.to(device, dtype=torch.float)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        model.eval()
        val_loss = 0.0
        dice_total = 0.0
        precision_total = 0.0
        recall_total = 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device, dtype=torch.float)
                masks = masks.to(device, dtype=torch.float)
                outputs = model(images)
                loss = loss_fn(outputs, masks)
                val_loss += loss.item()

                dice_total += dice_metric(outputs, masks).item()
                precision_total += precision_metric(outputs, masks).item()
                recall_total += recall_metric(outputs, masks).item()

        elapsed = time.time() - start_time

        avg_train_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_dice = dice_total / len(val_loader)
        avg_precision = precision_total / len(val_loader)
        avg_recall = recall_total / len(val_loader)

        with open(metrics_path, mode='a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow([epoch + 1, avg_train_loss, avg_val_loss, avg_dice, avg_precision, avg_recall, elapsed, batch_size, learning_rate])

        wandb.log({
            "Train Loss": avg_train_loss,
            "Val Loss": avg_val_loss,
            "Dice": avg_dice,
            "Precision": avg_precision,
            "Recall": avg_recall,
            "Time (s)": elapsed,
            "Learning Rate": learning_rate,
            "Batch Size": batch_size,
            "Epoch": epoch + 1
        })

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        dice_scores.append(avg_dice)
        precision_scores.append(avg_precision)
        recall_scores.append(avg_recall)

        # Graficar pérdidas y métricas
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss', color='tab:blue')
        ax1.plot(range(1, epoch+2), train_losses, label='Train Loss', color='tab:blue')
        ax1.plot(range(1, epoch+2), val_losses, label='Validation Loss', color='tab:red')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Scores', color='tab:green')
        ax2.plot(range(1, epoch+2), dice_scores, label='Dice', color='tab:pink')
        ax2.plot(range(1, epoch+2), precision_scores, label='Precision', color='tab:orange')
        ax2.plot(range(1, epoch+2), recall_scores, label='Recall', color='tab:olive')
        ax2.tick_params(axis='y', labelcolor='tab:green')

        fig.tight_layout()
        fig.legend(loc='upper right')
        fig_title = f"{os.path.splitext(os.path.basename(metrics_path))[0]}.png"
        fig_path = os.path.join("metrics", "grafics", fig_title)
        plt.savefig(fig_path)
        plt.close()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"Mejor modelo guardado en época {epoch+1} con val_loss: {best_val_loss:.4f}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"Sin mejora en {epochs_without_improvement} época(s)")
            if epochs_without_improvement >= patience:
                print(f"Early stopping activado en la época {epoch+1}")
                break

        torch.save(model.state_dict(), model_path)
        print(f"Último modelo guardado como {model_path}")

    torch.save(model.state_dict(), model_path)
    print(f"Último modelo guardado como {model_path}")

    wandb.finish()
    return model
