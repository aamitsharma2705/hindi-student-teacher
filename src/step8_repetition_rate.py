# ============================================================
# STEP 8.3 — REPETITION RATE MEASUREMENT (HINDI)
# ============================================================

import torch
import sentencepiece as spm
from transformers import GPT2LMHeadModel
from pathlib import Path

# ----------------------------
# DEVICE
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# PATHS
# ----------------------------
TOKENIZER_PATH = "tokenizers/hindi_bpe.model"
TRAINED_MODEL_PATH = "student_hindi_model"
UNTRAINED_MODEL_PATH = "student_hindi_untrained"

# ----------------------------
# LOAD TOKENIZER
# ----------------------------
sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_PATH)

# ----------------------------
# LOAD MODELS
# ----------------------------
trained_model = GPT2LMHeadModel.from_pretrained(TRAINED_MODEL_PATH)
untrained_model = GPT2LMHeadModel.from_pretrained(UNTRAINED_MODEL_PATH)

trained_model.to(device).eval()
untrained_model.to(device).eval()

# ----------------------------
# GENERATION FUNCTION
# ----------------------------
def generate_text(model, prompt, max_new_tokens=60):
    input_ids = torch.tensor(
        [sp.encode(prompt, out_type=int)],
        device=device
    )

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return sp.decode(input_ids[0].tolist())

# ----------------------------
# REPETITION METRIC
# ----------------------------
def repetition_rate(text, n=3):
    tokens = text.split()
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    return 1 - len(set(ngrams)) / len(ngrams)

# ----------------------------
# RUN COMPARISON
# ----------------------------
prompt = "भाषा मॉडल टेक्स्ट कैसे"

print("\nPROMPT:")
print(prompt)

untrained_text = generate_text(untrained_model, prompt)
trained_text = generate_text(trained_model, prompt)

print("\n--- UNTRAINED OUTPUT ---\n")
print(untrained_text)

print("\n--- TRAINED OUTPUT ---\n")
print(trained_text)

untrained_rep = repetition_rate(untrained_text)
trained_rep = repetition_rate(trained_text)

print("\nRepetition Rate (lower is better):")
print(f"Untrained: {untrained_rep:.3f}")
print(f"Trained  : {trained_rep:.3f}")
