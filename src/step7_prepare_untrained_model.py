import sentencepiece as spm
from transformers import GPT2Config, GPT2LMHeadModel
from pathlib import Path

# Load Hindi SentencePiece tokenizer
sp = spm.SentencePieceProcessor()
sp.load("tokenizers/hindi_bpe.model")

vocab_size = sp.get_piece_size()

# Same architecture as trained model
config = GPT2Config(
    vocab_size=vocab_size,
    n_embd=256,
    n_layer=4,
    n_head=4,
    n_positions=256
)

model = GPT2LMHeadModel(config)

# Save untrained model
SAVE_DIR = Path("student_hindi_untrained")
SAVE_DIR.mkdir(exist_ok=True)

model.save_pretrained(SAVE_DIR)

print("✅ Untrained Hindi model saved at student_hindi_untrained/")
