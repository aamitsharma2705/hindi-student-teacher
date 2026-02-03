# ============================================================
# STEP 6 — AUTOREGRESSIVE TEXT GENERATION (HINDI)
# Uses trained Hindi GPT-style model + SentencePiece tokenizer
# ============================================================

import torch
import sentencepiece as spm
from transformers import GPT2LMHeadModel
from pathlib import Path

# ============================================================
# 1. DEVICE SETUP
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# 2. LOAD SENTENCEPIECE TOKENIZER (HINDI)
# ============================================================

SP_MODEL_PATH = "tokenizers/hindi_bpe.model"

sp = spm.SentencePieceProcessor()
sp.load(SP_MODEL_PATH)

vocab_size = sp.get_piece_size()
print("Hindi vocab size:", vocab_size)

# ============================================================
# 3. LOAD TRAINED STUDENT MODEL
# ============================================================

MODEL_PATH = "student_hindi_model_trained_on_100000"

model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

# ============================================================
# 4. AUTOREGRESSIVE GENERATION FUNCTION
# ============================================================

def generate_hindi_text(
    prompt,
    max_new_tokens=50,
    temperature=0.3
):
    """
    Generates Hindi text token-by-token using
    autoregressive next-token prediction.
    """

    # Encode prompt → token IDs
    input_ids = torch.tensor(
        [sp.encode(prompt, out_type=int)],
        device=device
    )

    generated_ids = input_ids.clone()

    print("\nToken-by-token generation:\n")

    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids=generated_ids)
            logits = outputs.logits  # (1, seq_len, vocab_size)

        # Last-token prediction
        next_token_logits = logits[:, -1, :] / temperature
        probs = torch.softmax(next_token_logits, dim=-1)

        # Discrete next-token classification
        next_token_id = torch.multinomial(probs, num_samples=1)

        token_id = next_token_id.item()
        token_str = sp.decode([token_id])

        print(f"Step {step+1}: {token_str}")

        # Append token
        generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

    # Decode full sequence
    final_text = sp.decode(generated_ids[0].tolist())
    return final_text

# ============================================================
# 5. RUN GENERATION
# ============================================================

if __name__ == "__main__":

    prompt = "आप कैसे हैं"

    print("\nPROMPT:")
    print(prompt)

    generated_text = generate_hindi_text(
        prompt=prompt,
        max_new_tokens=50,
        temperature=1.0
    )

    print("\nFINAL GENERATED TEXT:\n")
    print(generated_text)

    # Optional: save output
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    (output_dir / "generated_text.txt").write_text(
        generated_text,
        encoding="utf-8"
    )

    print("\n✅ Output saved to outputs/generated_text.txt")
