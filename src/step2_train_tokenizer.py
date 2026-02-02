import os
import sentencepiece as spm

TRAIN_DIR = "data/train"

# Collect all .txt files explicitly
files = [
    os.path.join(TRAIN_DIR, f)
    for f in os.listdir(TRAIN_DIR)
    if f.endswith(".txt")
]

assert len(files) > 0, "❌ No training files found!"

file_list = ",".join(files)

print(f"✅ Found {len(files)} training files")

spm.SentencePieceTrainer.train(
    input=file_list,
    model_prefix="tokenizers/hindi_bpe",
    vocab_size=16000,
    model_type="bpe",
    character_coverage=0.9995,
    unk_id=0,
    bos_id=1,
    eos_id=2,
    pad_id=3
)

print("✅ Tokenizer trained successfully.")
