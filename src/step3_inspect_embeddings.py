import torch
import sentencepiece as spm
import torch.nn.functional as F

sp = spm.SentencePieceProcessor()
sp.load("tokenizers/hindi_bpe.model")

vocab_size = sp.get_piece_size()
embedding_dim = 256

embedding = torch.nn.Embedding(vocab_size, embedding_dim)

# Pick two tokens
token_a = sp.piece_to_id("है")
token_b = sp.piece_to_id("था")

vec_a = embedding(torch.tensor(token_a))
vec_b = embedding(torch.tensor(token_b))

cos_sim = F.cosine_similarity(vec_a, vec_b, dim=0)

print("Token A:", "है")
print("Token B:", "था")
print("Cosine similarity (random init):", cos_sim.item())
