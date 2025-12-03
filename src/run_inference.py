# src/run_inference.py

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .config import EMOTIONS, MODEL_NAME, MAX_LENGTH

def load_model(model_dir):
    # Load tokenizer for the chosen model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Ensure we have a pad token
    if tokenizer.pad_token is None:
        # fall back to eos or cls if needed
        if hasattr(tokenizer, "eos_token") and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.pad_token = tokenizer.cls_token

    # Load the classification model (HF hub or local dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    # Make sure model has a pad token id
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

def predict_segments(segments, tokenizer, model, device):
    results = []
    for idx, text in enumerate(segments, start=1):
        enc = tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = model(**enc)
            logits = outputs.logits[0]
            probs = F.softmax(logits, dim=-1).cpu().numpy()

        label_idx = probs.argmax()
        results.append({
            "segment_index": idx,
            "text": text,
            "logits": logits.cpu().numpy().tolist(),
            "probs": probs.tolist(),
            "label": EMOTIONS[label_idx]
        })
    return results
