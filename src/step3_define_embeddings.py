import torch
import sentencepiece as spm

# Load tokenizer
sp = spm.SentencePieceProcessor()
sp.load("tokenizers/hindi_bpe.model")

vocab_size = sp.get_piece_size()
embedding_dim = 256  # design choice (moderate, exam-safe)

print(f"Vocabulary size: {vocab_size}")
print(f"Embedding dimension: {embedding_dim}")

# Define embedding layer
embedding = torch.nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embedding_dim
)

# Example token sequence
sample_text = "यह एक उदाहरण वाक्य है"
token_ids = sp.encode(sample_text, out_type=int)

token_tensor = torch.tensor(token_ids, dtype=torch.long)

# Lookup embeddings
embedded_tokens = embedding(token_tensor)

print("Token IDs:", token_ids)
print("Embedded shape:", embedded_tokens.shape)
