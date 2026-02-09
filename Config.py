from pathlib import Path

class Config:
    TEXT_MODEL_NAME = "bert-base-uncased"
    IMAGE_MODEL_NAME = "tf_efficientnet_b0"

    BATCH_SIZE = 16
    TEXT_LR = 3e-5
    IMAGE_LR = 1e-4
    HEAD_LR = 5e-4
    EPOCHS = 5
    DROPOUT = 0.1
    EMB_DIM = 256

    # Пути
    try:
        BASE_DIR = Path(__file__).parent  # если запускаем как скрипт
    except NameError:
        BASE_DIR = Path.cwd()  # если ноутбук
    DATA_CSV = BASE_DIR / "data" / "dish.csv"
    INGR_CSV = BASE_DIR / "data" / "ingredients.csv"
    IMAGE_ROOT = BASE_DIR / "data" / "images"
    SAVE_PATH = BASE_DIR / "best_model.pth"

    MAE_THRESHOLD = 50.0
