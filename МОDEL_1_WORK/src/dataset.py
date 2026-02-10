import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from transformers import AutoTokenizer

# -------------------------
# Подключение Config
# -------------------------
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from Config import Config

# -------------------------
# Ingredient Groups
# -------------------------
GROUPS = {
    "meat": ["beef","chicken","pork","lamb","turkey","fish","salmon","tuna","bacon","ham","steak"],
    "vegetable": ["lettuce","spinach","broccoli","tomato","carrot","onion","pepper","zucchini","cabbage"],
    "fruit": ["apple","banana","orange","berry","berries","melon","pear","peach"],
    "grain": ["rice","pasta","bread","oats","noodles","quinoa"],
    "sauce": ["sauce","ketchup","mayonnaise","dressing","oil","vinegar"],
    "drink": ["juice","milk","smoothie","coffee","tea"],
    "water": ["water"]
}

def detect_group(ingr):
    ingr = ingr.lower()
    for g, keys in GROUPS.items():
        for k in keys:
            if k in ingr:
                return g
    return "other"

GROUP2ID = {g:i for i,g in enumerate(GROUPS.keys())}
GROUP2ID["other"] = len(GROUP2ID)


    
def get_transforms(config, ds_type="train"):
    """
    Возвращает трансформации для датасета.
    train -> Resize + Normalize + ToTensorV2 (можно добавить аугментации)
    test  -> Resize + Normalize + ToTensorV2
    """
    if ds_type == "train":
        return A.Compose([
            A.Resize(224, 224),
            # Можно добавить аугментации, например:
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(0.2, 0.2, 0.2, 0.1),
            A.Normalize(),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(),
            ToTensorV2()
        ])


# -------------------------
# Dataset Class
# -------------------------
class MultimodalDataset(Dataset):
    def __init__(self, df, transforms, ingr_cal):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.tokenizer = AutoTokenizer.from_pretrained(Config.TEXT_MODEL_NAME)
        self.ingr_cal = ingr_cal

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.loc[idx]

        text = r["ingredients"]

        # ===== TARGET =====
        label = torch.tensor(r["cal_per_100g"], dtype=torch.float32)

        # ===== KNOWN INGREDIENT SIGNAL =====
        known_cal = torch.tensor(
            known_ingr_cal(text, self.ingr_cal),
            dtype=torch.float32
        )
        known_mask = torch.tensor(
            known_ingr_mask(text, self.ingr_cal),
            dtype=torch.float32
        )

        # ===== GROUP =====
        group_id = torch.tensor(GROUP2ID[detect_group(text)], dtype=torch.long)

        # ===== IMAGE =====
        img_path = os.path.join(Config.IMAGE_ROOT, str(r["dish_id"]), "rgb.png")
        img = Image.open(img_path).convert("RGB")
        img = self.transforms(image=np.array(img))["image"]

        # ===== MASS =====
        mass = torch.tensor(r["total_mass"], dtype=torch.float32)

        return {
            "text": text,
            "image": img,
            "label": label,                 # kcal / 100g
            "known_cal": known_cal,         # kcal / 100g (если 1 ингредиент)
            "known_mask": known_mask,       # 1.0 если можно доверять
            "group": group_id,
            "mass": mass
        }


def collate_fn(batch, tokenizer):
    texts = [b["text"] for b in batch]
    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "image": torch.stack([b["image"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "known_cal": torch.stack([b["known_cal"] for b in batch]),
        "known_mask": torch.stack([b["known_mask"] for b in batch]),
        "group": torch.stack([b["group"] for b in batch]),
        "mass": torch.stack([b["mass"] for b in batch]),
    }




def known_ingr_cal(ingr_str, ingr_cal):
    ids = ingr_str.split(";")

    # РОВНО 1 ингредиент и он известен
    if len(ids) == 1 and ids[0] in ingr_cal:
        return float(ingr_cal[ids[0]])

    # иначе ничего не подсказываем
    return 0.0


def known_ingr_mask(ingr_str, ingr_cal):
    ids = ingr_str.split(";")
    return float(len(ids) == 1 and ids[0] in ingr_cal)



df_all = pd.read_csv(Config.DATA_CSV)
df_train = df_all[df_all["split"] == "train"].reset_index(drop=True)
df_test  = df_all[df_all["split"] == "test"].reset_index(drop=True)
df_train["cal_per_100g"] = df_train["total_calories"] / df_train["total_mass"] * 100
df_test["cal_per_100g"]  = df_test["total_calories"] / df_test["total_mass"] * 100



# -------------------------
# Build Ingredient Cal Dict
# -------------------------
single = df_train[df_train["ingredients"].str.count(";") == 0].copy()
single["cal_per_g"] = single["total_calories"] / single["total_mass"]
ingr_cal = {r["ingredients"]: r["cal_per_g"] for _, r in single.iterrows()}
# -------------------------
# MAIN
# -------------------------
if __name__ == "__main__":
    print("=== DATASET SANITY CHECK ===")

    print("CSV path:", Config.DATA_CSV)
    print("Exists:", os.path.exists(Config.DATA_CSV))

    # ---------- трансформации ----------
    transforms = get_transforms(Config, ds_type="train")

    # ---------- создаём датасет КОРРЕКТНО ----------
    dataset = MultimodalDataset(
        df_train,
        transforms,
        ingr_cal=ingr_cal
    )

    print("Dataset length:", len(dataset))

    # ---------- проверяем несколько сэмплов ----------
    for idx in [0, 1, 2]:
        sample = dataset[idx]

        print(f"\n--- SAMPLE {idx} ---")
        print("ingredients:", sample["text"])
        print("label (kcal/100g):", sample["label"].item())
        print("known_cal:", sample["known_cal"].item())
        print("known_mask:", sample["known_mask"].item())
        print("group id:", sample["group"].item())
        print("mass (g):", sample["mass"].item())
        print("image shape:", sample["image"].shape)

        # 🔍 логическая проверка
        ids = sample["text"].split(";")
        if len(ids) == 1 and ids[0] in ingr_cal:
            assert sample["known_mask"].item() == 1.0, "known_mask должен быть 1"
            assert sample["known_cal"].item() > 0, "known_cal должен быть > 0"
        else:
            assert sample["known_mask"].item() == 0.0, "known_mask должен быть 0"
            assert sample["known_cal"].item() == 0.0, "known_cal должен быть 0"

    print("\n✅ Single-sample checks passed")

    # ---------- проверка DataLoader + collate_fn ----------
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, dataset.tokenizer)
    )

    batch = next(iter(loader))

    print("\n=== BATCH CHECK ===")
    for k, v in batch.items():
        print(k, "→", v.shape)

    print("\nknown_mask in batch:", batch["known_mask"])
    print("known_cal in batch:", batch["known_cal"])

    print("\n✅ Dataset + collate_fn are WORKING CORRECTLY")




