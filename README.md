# CeraMIRScan: Mid-infrared OCT Scan Dataset for Ceramic Quality Assessment

This repository supports the research work **_“CeraMIRScan: Mid-infrared OCT Scan Dataset for Ceramic Quality Assessment”_**, providing all tools necessary to **reproduce and extend defect segmentation experiments** on ceramic components.

Our aim is to enable researchers and industry professionals to explore **Non-Destructive Testing (NDT)** using **Mid-Infrared Optical Coherence Tomography (MIR-OCT)** combined with **Deep Learning** for defect detection and segmentation.

---

## 1. Installation & Dependencies

**Requirements**  
- Python **≥ 3.8**

Install required dependencies:

    pip install -r requirements.txt

## 2. Preparing the Dataset Structure

The original CeraMIRScan dataset is organized by volumes (each volume contains a full MIR-OCT scan).
However, the training code in this repository expects the splits with volumes inside images/ and masks/.

- Download the volumetric MIR-OCT scans of ceramic parts with pixel-level defect annotations (link)
- The expected dataset structure is as follows:
```
data/
└── BD_2D/
    ├── train/
    │   ├── images/
    │   │   ├── A23652_SCAN1_S1/
    │   │   ├── A23652_SCAN2_S1/
    │   └── mask/
    │       ├── A23652_SCAN1_S1/
    │       ├── A23652_SCAN2_S1/
    ├── val/
    │   ├── images/
    │   │   ├── A23654_SCAN1_S1/
    │   │   ├── Pilot-Demo_x-mode_images_2/
    │   └── mask/
    │       ├── A23654_SCAN1_S1/
    │       ├── Pilot-Demo_x-mode_images_2/
```
Notes:
- Each image must have a corresponding mask with the exact same filename.
- Masks are binary (0 = background, 1 = defect).



## 3. Training the model

To train the model, run:

    python main.py --mode train 

Inside `main.py`, you can manually configure the main hyperparameters before training:

- `batch_size`: e.g. batch_size = 4  
- `learning_rate`: e.g. learning_rate = 0.001  
- `num_epochs`: e.g. num_epochs = 100  
- `patience`: early stopping patience, e.g. patience = 20  

The trained model will be saved automatically in:

    /trains/n_train_day_month_year.pth


## 4. Making predictions

After training, you can generate predictions by running:

    python main.py --mode predict

Inside `main.py`, you must set the following variables:

- `model_filename`: the name of the trained model file to use, e.g.  
    model_filename = '1_train_23-07-25.pth'

- `nombre_volumen_prediccion`: the name of the volume folder inside `data/BD_2D/test/test_images/`, e.g.  
    nombre_volumen_prediccion = "A23654_SCAN2_S1"

Predicted masks will be saved in a new folder:

    predictions/A23654_SCAN2_S1/

In addition, a CSV file with evaluation metrics for that prediction is automatically generated in the same output folder:

    predictions/A23654_SCAN2_S1/metrics.csv



## 5. Metric tracking

The training and validation process uses [Weights & Biases (wandb)](https://wandb.ai/) to track loss, metrics, and training progress.

To use wandb, make sure to log in with:

    wandb login

You can also configure a project name and entity to store your results online.

Additionally, the training metrics are also saved locally in:

    /metrics/n_train.csv

This CSV file contains per-epoch training and validation results, including loss, Dice score, precision, and recall.

The following plots are also generated and saved in the `/metrics/` folder:
- Training and validation loss curves
- Training and validation metric curves (Dice, Precision, Recall)

