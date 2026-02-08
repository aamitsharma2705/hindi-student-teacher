import torch
import torch.nn.functional as F
import sentencepiece as spm
from transformers import (
    BertForMaskedLM,
    BertTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer
)

# -------------------------------
# 1️⃣ Load Tokenizer
# -------------------------------
# Teacher: BERT (frozen)
teacher_name = "bert-base-uncased"

teacher_tokenizer = BertTokenizer.from_pretrained(teacher_name)
teacher_model = BertForMaskedLM.from_pretrained(teacher_name)

teacher_model.eval()
for param in teacher_model.parameters():
    param.requires_grad = False


# Student: DistilGPT-2
student_name = "distilgpt2"

student_tokenizer = GPT2Tokenizer.from_pretrained(student_name)
student_model = GPT2LMHeadModel.from_pretrained(student_name)

student_model.train()

# Hindi tokenizer (ours) — completely separate vocab + model type
hindi_sp = spm.SentencePieceProcessor()
hindi_sp.load("tokenizers/hindi_bpe.model")
hindi_vocab_size = hindi_sp.get_piece_size()

text = "The cat sat on the mat"
hindi_text = "यह एक उदाहरण वाक्य है"

# Student input
student_inputs = student_tokenizer(
    text,
    return_tensors="pt"
)

# Teacher input (same text, but tokenized separately)
teacher_inputs = teacher_tokenizer(
    text,
    return_tensors="pt"
)

# -------------------------------
# 🔎 Incompatibility showcase
# -------------------------------
print("\n--- Tokenizer mismatch (English models vs Hindi SP) ---")
print("Teacher tokenizer vocab size:", teacher_tokenizer.vocab_size)
print("Student tokenizer vocab size:", student_tokenizer.vocab_size)
print("Hindi SentencePiece vocab size:", hindi_vocab_size)

teacher_hi_tokens = teacher_tokenizer.tokenize(hindi_text)
student_hi_tokens = student_tokenizer.tokenize(hindi_text)
hindi_pieces = hindi_sp.encode(hindi_text, out_type=str)

print("\nHindi text:", hindi_text)
print("Teacher tokens (BERT, WordPiece):", teacher_hi_tokens)
print("Student tokens (GPT-2 BPE):", student_hi_tokens)
print("Hindi SP pieces (intended):", hindi_pieces[:12], "...")
print(
    "Tokenizer mismatch: teacher/student cannot express Hindi pieces; "
    "they will degrade to [UNK]/byte fallbacks."
)

print("\n--- Vocabulary misalignment ---")
print(
    "Teacher/Student token ids are unrelated to Hindi SP ids; "
    "there is no mapping between vocabularies."
)

print("\n--- Mechanism obscuration ---")
print(
    "Teacher is masked-LM (bidirectional, uses [MASK]); "
    "student is autoregressive (causal). "
    "Different objectives + tokenizers -> distillation signals are noisy."
)

student_outputs = student_model(**student_inputs)
student_logits = student_outputs.logits  # [batch, seq_len, vocab_size]


# Copy input ids and mask one token
input_ids = teacher_inputs["input_ids"].clone()
mask_index = 3  # arbitrary position
input_ids[0, mask_index] = teacher_tokenizer.mask_token_id

teacher_outputs = teacher_model(
    input_ids=input_ids,
    attention_mask=teacher_inputs["attention_mask"]
)

teacher_logits = teacher_outputs.logits  # [batch, seq_len, vocab_size]

# Select the masked position
teacher_logits_masked = teacher_logits[:, mask_index, :]
student_logits_masked = student_logits[:, mask_index, :]


temperature = 2.0
alpha = 0.5  # balance between hard and soft loss


# Ground truth = next token
labels = student_inputs["input_ids"][:, mask_index]

ce_loss = F.cross_entropy(
    student_logits_masked,
    labels
)


# Get ground-truth token id for student
gt_token_id = labels.item()

# Teacher probability for that token (if exists, else ignore)
with torch.no_grad():
    teacher_probs_full = F.softmax(teacher_logits_masked / temperature, dim=-1)

# NOTE: We cannot index teacher by GPT token id safely
# So we distill confidence, not token identity

teacher_confidence = teacher_probs_full.max(dim=-1).values
student_confidence = F.softmax(student_logits_masked / temperature, dim=-1).max(dim=-1).values

# KL-style loss on confidence scalars
distill_loss = F.mse_loss(student_confidence, teacher_confidence)


total_loss = alpha * ce_loss + (1 - alpha) * distill_loss

print("Cross-Entropy Loss:", ce_loss.item())
print("Distillation Loss (confidence):", distill_loss.item())
print("Total Loss:", total_loss.item())


