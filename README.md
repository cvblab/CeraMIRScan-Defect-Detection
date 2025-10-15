1. DEPENDENCIES INSTALLATION

Requirements: Python >= 3.8

To install all required packages, run:

    pip install -r requirements.txt

Contents of the `requirements.txt` file:

    segmentation-models-pytorch
    albumentations
    opencv-python-headless==4.7.0.72
    opencv-fixer==0.2.5
    pytz
    seaborn
    wandb
    PyMuPDF
    scikit-image


2. FOLDER STRUCTURE

The expected dataset structure is as follows:

    data/
    └── BD_2D/
        ├── train/
        │   ├── images/
        │   │   ├── volume1_0.png
        │   │   ├── volume1_1.png
        │   └── mask/
        │       ├── volume1_0.png
        │       ├── volume1_1.png
        ├── val/
        │   ├── images/
        │   └── mask/
        └── test/
            ├── test_images/
            └── test_mask/

Notes:
- Each image must have a corresponding mask with the exact same filename.
- Masks are binary (0 = background, 1 = defect).
- `train/`, `val/`, and `test/` sets must be properly matched.



3. TRAINING THE MODEL

To train the model, run:

    python main.py --mode train 

Inside `main.py`, you can manually configure the main hyperparameters before training:

- `batch_size`: e.g. batch_size = 4  
- `learning_rate`: e.g. learning_rate = 0.001  
- `num_epochs`: e.g. num_epochs = 100  
- `patience`: early stopping patience, e.g. patience = 20  

The trained model will be saved automatically in:

    /trains/n_train_day_month_year.pth



4. MAKING PREDICTIONS

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



5. METRIC TRACKING

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
