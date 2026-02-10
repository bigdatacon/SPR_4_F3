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
    def __init__(self, fusion_type="concat"):
        """
        fusion_type: "concat" | "multiply" | "cross_attention"
        """
        super().__init__()
        self.fusion_type = fusion_type

        self.text = AutoModel.from_pretrained(Config.TEXT_MODEL_NAME)
        self.image = timm.create_model(Config.IMAGE_MODEL_NAME, pretrained=True, num_classes=0)
        self.group_emb = nn.Embedding(len(GROUP2ID), 8)

        text_dim = self.text.config.hidden_size
        img_dim = self.image.num_features
        group_dim = 8
        extra_dim = 1  # avg_cal

        if fusion_type == "concat":
            in_dim = text_dim + img_dim + extra_dim + group_dim

        elif fusion_type == "multiply":
            # проекция изображения в размерность текста
            self.i_proj = nn.Linear(img_dim, text_dim)
            in_dim = text_dim + extra_dim + group_dim

        elif fusion_type == "cross_attention":
            # Cross-attention: текст как query, изображение как key/value
            self.t_proj = nn.Linear(text_dim, 256)
            self.i_proj = nn.Linear(img_dim, 256)
            self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=4, batch_first=True)
            in_dim = 256 + extra_dim + group_dim

        else:
            raise ValueError("Unknown fusion_type")

        self.head = nn.Sequential(
            nn.Linear(in_dim, Config.EMB_DIM),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(Config.EMB_DIM, 1)
        )

    def forward(self, input_ids, attention_mask, image, avg_cal, group):
        t = self.text(input_ids, attention_mask).pooler_output  
        i = self.image(image)                                    
        g = self.group_emb(group)                                 
        avg_cal = avg_cal.unsqueeze(1)                            

        if self.fusion_type == "concat":
            x = torch.cat([t, i, avg_cal, g], dim=1)

        elif self.fusion_type == "multiply":
            i_proj = self.i_proj(i)
            x = t * i_proj  
            x = torch.cat([x, avg_cal, g], dim=1)

        elif self.fusion_type == "cross_attention":
            t_proj = self.t_proj(t).unsqueeze(1)   # [B, 1, 256]
            i_proj = self.i_proj(i).unsqueeze(1)   # [B, 1, 256]
            attn_out, _ = self.attention(query=t_proj, key=i_proj, value=i_proj)
            x = torch.cat([attn_out.squeeze(1), avg_cal, g], dim=1)

        out = self.head(x).squeeze(1)

        # ===== CLIP ограничения - то есть руками ограничиваем максимум на 100 г исходя из средней калорийности продуктов в реальной жизни=====
        for g_name, clip_val in {"meat": 600, "grain": 400, "vegetable": 150,
                                 "fruit": 200, "sauce": 30, "water": 0, "drink": 300}.items():
            mask = (group == GROUP2ID.get(g_name, -1))
            if mask.any():
                out[mask] = out[mask].clamp(max=clip_val)

        return out



if __name__ == "__main__":
    # Тест модели на случайных данных
    model = MultimodalModel()
    input_ids = torch.randint(0, 1000, (2, 16))
    attention_mask = torch.ones_like(input_ids)
    image = torch.randn(2, 3, 224, 224)
    avg_cal = torch.tensor([200.0, 300.0])
    group = torch.tensor([GROUP2ID["meat"], GROUP2ID["vegetable"]])
    output = model(input_ids, attention_mask, image, avg_cal, group)
    print("Output shape:", output.shape)