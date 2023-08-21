import transformers
from datasets import load_dataset
import lightning.pytorch as pl
from transformers import SamModel, SamProcessor

dataset = load_dataset("nielsr/breast-cancer", split="train")
processor = SamProcessor.from_pretrained("wanglab/medsam-vit-base")
model = SamModel.from_pretrained("wanglab/medsam-vit-base")
# Visua
example = dataset[0]
image = example["image"]
image

import numpy as np

idx = 10

# load image
image = dataset[idx]["image"]
image

