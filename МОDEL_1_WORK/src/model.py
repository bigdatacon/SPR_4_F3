import os
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from functools import partial
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import timm
from transformers import AutoTokenizer, AutoModel
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn

from dataset import GROUP2ID

# -------------------------
# Подключение Config
# -------------------------
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))
from Config import Config

class MultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.text = AutoModel.from_pretrained(Config.TEXT_MODEL_NAME)
        self.image = timm.create_model(
            Config.IMAGE_MODEL_NAME,
            pretrained=True,
            num_classes=0
        )
        self.group_emb = nn.Embedding(len(GROUP2ID), 8)

        text_dim = self.text.config.hidden_size
        img_dim = self.image.num_features
        group_dim = 8

        # known_cal * known_mask → 1 фича
        in_dim = text_dim + img_dim + group_dim + 1

        self.head = nn.Sequential(
            nn.Linear(in_dim, Config.EMB_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.EMB_DIM, 1)
        )

    def forward(
        self,
        input_ids,
        attention_mask,
        image,
        known_cal,
        known_mask,
        group
    ):
        t = self.text(input_ids, attention_mask).pooler_output
        i = self.image(image)
        g = self.group_emb(group)

        # 🔑 КЛЮЧЕВОЙ МОМЕНТ
        known_signal = (known_cal * known_mask).unsqueeze(1)

        x = torch.cat([t, i, g, known_signal], dim=1)
        out = self.head(x).squeeze(1)

        # ===== CLIP по категориям (kcal / 100g) =====
        CLIP_RULES = {
            "meat": 600,
            "grain": 400,
            "vegetable": 150,
            "fruit": 200,
            "sauce": 30,
            "water": 0,
            "drink": 300,
        }

        for g_name, clip_val in CLIP_RULES.items():
            mask = (group == GROUP2ID.get(g_name, -1))
            if mask.any():
                out[mask] = out[mask].clamp(max=clip_val)

        return out



if __name__ == "__main__":
    model = MultimodalModel()

    batch_size = 2
    seq_len = 16

    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    image = torch.randn(batch_size, 3, 224, 224)

    known_cal = torch.tensor([52.0, 0.0])      # kcal/100g
    known_mask = torch.tensor([1.0, 0.0])      # 1 если 1 ингредиент
    group = torch.tensor([
        GROUP2ID["meat"],
        GROUP2ID["vegetable"]
    ])

    output = model(
        input_ids,
        attention_mask,
        image,
        known_cal,
        known_mask,
        group
    )

    print("Output shape:", output.shape)
    print("Output:", output)
