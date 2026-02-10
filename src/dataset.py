import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
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
    cfg = timm.get_pretrained_cfg(config.IMAGE_MODEL_NAME)
    if ds_type == "train":
        return A.Compose([
            A.Resize(224, 224),

            A.RandomCrop(cfg.input_size[1], cfg.input_size[2]),  #добавил
            A.ColorJitter(0.2,0.2,0.2,0.1),

            A.HorizontalFlip(p=0.5),
            A.ColorJitter(0.2, 0.2, 0.2, 0.1),
            A.Normalize(),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.SmallestMaxSize(max_size=max(cfg.input_size[1], cfg.input_size[2])),  #добавил
            A.CenterCrop(cfg.input_size[1], cfg.input_size[2]),

            A.Resize(224, 224),
            A.Normalize(),
            ToTensorV2()
        ])


class MultimodalDataset(Dataset):
    def __init__(self, df, transforms):
        self.df = df.reset_index(drop=True)
        self.tokenizer = AutoTokenizer.from_pretrained(Config.TEXT_MODEL_NAME)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.loc[idx]
        text = r["ingredients"]
        label = torch.tensor(r["cal_per_100g"], dtype=torch.float32)
        avg_cal = torch.tensor(r["avg_ingr_cal"], dtype=torch.float32)
        group_id = torch.tensor(GROUP2ID[detect_group(text)], dtype=torch.long)

        img_path = os.path.join(Config.IMAGE_ROOT, str(r['dish_id']), "rgb.png")
        img = Image.open(img_path).convert("RGB")
        img = self.transforms(image=np.array(img))["image"]
        mass = torch.tensor(r["total_mass"], dtype=torch.float32)

        # return {
        #     "text": text,
        #     "image": img,
        #     "label": label,
        #     "avg_cal": avg_cal,
        #     "group": group_id,
        #     "mass": mass
        # }

        return {
            "dish_id": r["dish_id"],
            "text": text,
            "image": img,
            "label": label,
            "avg_cal": avg_cal,
            "group": group_id,
            "mass": mass
        }





def collate_fn(batch, tokenizer):
    texts = [b["text"] for b in batch]
    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    # result = {
    #     "input_ids": tokens["input_ids"],
    #     "attention_mask": tokens["attention_mask"],
    #     "image": torch.stack([b["image"] for b in batch]),
    #     "label": torch.stack([b["label"] for b in batch]),
    #     "avg_cal": torch.stack([b["avg_cal"] for b in batch]),
    #     "group": torch.stack([b["group"] for b in batch]),
    #     "mass": torch.stack([b["mass"] for b in batch])
    # }

    result = {
        "dish_id": [b["dish_id"] for b in batch],
        "text": [b["text"] for b in batch],
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "image": torch.stack([b["image"] for b in batch]),
        "label": torch.stack([b["label"] for b in batch]),
        "avg_cal": torch.stack([b["avg_cal"] for b in batch]),
        "group": torch.stack([b["group"] for b in batch]),
        "mass": torch.stack([b["mass"] for b in batch])
    }


    # print("Collate batch keys:", result.keys())  # ← для проверки
    return result


"Берем калорийность ингредиентов из блюд, в которых только 1 ингредиент, и усредняем по этому ингредиенту. Если ингредиента нет в этих блюдах, то определяем его группу и используем среднюю калорийность по группе."
def avg_ingr_cal(ingr_str, ingr_cal): 
    ids = ingr_str.split(";")
    cal_values = []
    for i in ids:
        if i in ingr_cal:
            cal_values.append(ingr_cal[i])
        else:
            group = detect_group(i)
            cal_values.append({
                "meat": 2.5,
                "grain": 1.3,
                "vegetable": 0.4,
                "fruit": 0.6,
                "sauce": 3.5,
                "water": 0.0
            }.get(group, 1.0))
    return torch.tensor(cal_values, dtype=torch.float32).mean().item()



df_all = pd.read_csv(Config.DATA_CSV)
df_train = df_all[df_all["split"] == "train"].reset_index(drop=True)
df_test  = df_all[df_all["split"] == "test"].reset_index(drop=True)


single = df_train[df_train["ingredients"].str.count(";") == 0].copy()
single["cal_per_g"] = single["total_calories"] / single["total_mass"]
ingr_cal = {r["ingredients"]: r["cal_per_g"] for _, r in single.iterrows()}

if __name__ == "__main__":
    # Проверяем CSV
    print("CSV path:", Config.DATA_CSV)
    print("Exists:", os.path.exists(Config.DATA_CSV))

    for df, ds_type in zip([df_train, df_test], ["train", "test"]):
        # df["avg_ingr_cal"] = df["ingredients"].apply(avg_ingr_cal)
        df["avg_ingr_cal"] = df["ingredients"].apply(lambda x: avg_ingr_cal(x, ingr_cal))
        df["cal_per_100g"] = df["total_calories"] / df["total_mass"] * 100

        transforms = get_transforms(Config, ds_type=ds_type)
        dataset = MultimodalDataset(df, transforms)
        sample = dataset[0]
        print(f"Sample from {ds_type} dataset:", sample)
        print(f" sample[mass] : {sample['mass']} ")  # масса блюда в 

        b = next(iter(df.iterrows()))[1]
        print(b["dish_id"][:2])
        print(b["ingredients"][:50])
        print("ok")





