import os
import sentencepiece as spm

TOKENIZER_MODEL = "tokenizers/hindi_bpe.model"

TRAIN_DIR = "data/train"
VALID_DIR = "data/valid"

OUTPUT_TRAIN = "data/train_tokenized.txt"
OUTPUT_VALID = "data/valid_tokenized.txt"

CHUNK_SIZE = 5000  # characters per chunk (safe default)

sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_MODEL)


def tokenize_folder_chunked(input_dir, output_file):
    with open(output_file, "w", encoding="utf-8") as out:
        for filename in os.listdir(input_dir):
            if not filename.endswith(".txt"):
                continue

            file_path = os.path.join(input_dir, filename)

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read().strip()

                if not text:
                    continue

                # Process text in chunks
                for i in range(0, len(text), CHUNK_SIZE):
                    chunk = text[i:i + CHUNK_SIZE]
                    token_ids = sp.encode(chunk, out_type=int)
                    out.write(" ".join(map(str, token_ids)) + "\n")


print("🔄 Tokenizing TRAIN split...")
tokenize_folder_chunked(TRAIN_DIR, OUTPUT_TRAIN)

print("🔄 Tokenizing VALIDATION split...")
tokenize_folder_chunked(VALID_DIR, OUTPUT_VALID)

print("✅ Chunked tokenization completed.")
