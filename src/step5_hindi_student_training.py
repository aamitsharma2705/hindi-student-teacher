# ============================================================
# STEP 5 — HINDI GPT-STYLE LANGUAGE MODEL TRAINING (SUBSET)
# Runs on Kaggle GPU if available, else CPU
# ============================================================

import torch
import sentencepiece as spm
from transformers import GPT2Config, GPT2LMHeadModel
from torch.optim import AdamW
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm


# from IPython.display import FileLink
# FileLink(r'folder/file')



# ============================================================
# 1. DEVICE SETUP (GPU IF AVAILABLE)
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))

# ============================================================
# 2. LOAD SENTENCEPIECE HINDI TOKENIZER
# ============================================================

SP_MODEL_PATH = "/kaggle/input/dataset/tokenizers/tokenizers/hindi_bpe.model"

sp = spm.SentencePieceProcessor()
sp.load(SP_MODEL_PATH)

vocab_size = sp.get_piece_size()
print("Hindi vocab size:", vocab_size)

# ============================================================
# 3. BUILD GPT-STYLE STUDENT MODEL (FROM SCRATCH)
# ============================================================

config = GPT2Config(
    vocab_size=vocab_size,
    n_embd=256,
    n_layer=4,
    n_head=4,
    n_positions=256
)

model = GPT2LMHeadModel(config)
model.to(device)
model.train()

optimizer = AdamW(model.parameters(), lr=3e-4)

# ============================================================
# 4. DATASET SELECTION (KAGGLE INPUT PATH)
# ============================================================

TRAIN_DIR = Path(
    "/kaggle/input/hindi-wikipedia-articles-172k/"
    "train/train"
)

all_files = sorted(TRAIN_DIR.glob("*.txt"))

# 🔒 START SMALL (CHANGE TO 10000 AFTER VERIFICATION)
NUM_FILES = 100000
train_files = all_files[:NUM_FILES]

print(f"Total files found: {len(all_files)}")
print(f"Training on files: {len(train_files)}")

# ============================================================
# 5. TRAINING HYPERPARAMETERS
# ============================================================

MAX_SEQ_LEN = 256
EPOCHS = 1        # 1 epoch is enough for setup verification

# ============================================================
# 6. HELPER: FILE → TOKEN CHUNKS (WITH BLANK CHECK)
# ============================================================

def file_to_chunks(path, max_len):
    """
    Reads a file, skips blank or very short files,
    and yields fixed-length token chunks.
    """
    text = path.read_text(encoding="utf-8", errors="ignore").strip()

    # Skip blank files
    if not text:
        return

    token_ids = sp.encode(text, out_type=int)

    # Skip very short files
    if len(token_ids) < max_len:
        return

    for i in range(0, len(token_ids) - max_len, max_len):
        yield token_ids[i:i + max_len]

# ============================================================
# 7. TRAINING LOOP
# ============================================================

total_steps = 0
skipped_files = 0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    for file_path in tqdm(train_files):
        produced_chunk = False

        for chunk in file_to_chunks(file_path, MAX_SEQ_LEN):
            produced_chunk = True

            input_ids = torch.tensor([chunk], device=device)

            # Next-token labels
            labels = input_ids.clone()
            labels[:, :-1] = input_ids[:, 1:]
            labels[:, -1] = -100

            outputs = model(input_ids=input_ids)
            loss = F.cross_entropy(
                outputs.logits.view(-1, vocab_size),
                labels.view(-1),
                ignore_index=-100
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_steps += 1

        if not produced_chunk:
            skipped_files += 1

    print(f"Epoch {epoch + 1} completed")

print("\nTraining finished")
print("Total training steps:", total_steps)
print("Skipped blank/short files:", skipped_files)

# ============================================================
# 8. SAVE CHECKPOINT (FOR STEP 6)
# ============================================================

SAVE_DIR = "student_hindi_model"
Path(SAVE_DIR).mkdir(exist_ok=True)

model.save_pretrained(SAVE_DIR)

print(f"\n✅ Hindi student model saved at: {SAVE_DIR}")
