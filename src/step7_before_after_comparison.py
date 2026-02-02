# ============================================================
# STEP 7 — BEFORE vs AFTER TRAINING COMPARISON (HINDI)
# ============================================================

import torch
import sentencepiece as spm
from transformers import GPT2LMHeadModel

# ----------------------------
# DEVICE
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------
# LOAD TOKENIZER
# ----------------------------
sp = spm.SentencePieceProcessor()
sp.load("tokenizers/hindi_bpe.model")

# ----------------------------
# LOAD MODELS
# ----------------------------
untrained_model = GPT2LMHeadModel.from_pretrained("student_hindi_untrained")
trained_model = GPT2LMHeadModel.from_pretrained("student_hindi_model")

untrained_model.to(device).eval()
trained_model.to(device).eval()

# ----------------------------
# GENERATION FUNCTION
# ----------------------------
def generate(model, prompt, max_new_tokens=50):
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
# PROMPT (SAME FOR BOTH)
# ----------------------------
prompt = "आप कैसे हैं"

print("\nPROMPT:")
print(prompt)

print("\n--- UNTRAINED MODEL OUTPUT ---\n")
print(generate(untrained_model, prompt))

print("\n--- TRAINED MODEL OUTPUT ---\n")
print(generate(trained_model, prompt))
