import torch
from transformers import AutoModelForSequenceClassification

MODEL_OPTIONS = {
    "roberta-base": {
        "batch_size": 10,
    },
    "distilbert-base-uncased": {
        "batch_size": 20
    }
}







