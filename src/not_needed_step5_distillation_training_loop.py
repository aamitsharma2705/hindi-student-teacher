import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import (
    BertForMaskedLM,
    BertTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer
)
from tqdm import tqdm
from pathlib import Path

# =====================================================
# DEVICE
# =====================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =====================================================
# TEACHER MODEL (FROZEN)
# =====================================================

teacher_name = "bert-base-uncased"

teacher_tokenizer = BertTokenizer.from_pretrained(teacher_name)
teacher_model = BertForMaskedLM.from_pretrained(teacher_name)

teacher_model.to(device)
teacher_model.eval()

for p in teacher_model.parameters():
    p.requires_grad = False

# =====================================================
# STUDENT MODEL (TRAINABLE)
# =====================================================

student_name = "distilgpt2"

student_tokenizer = GPT2Tokenizer.from_pretrained(student_name)
student_model = GPT2LMHeadModel.from_pretrained(student_name)

student_model.to(device)
student_model.train()

# GPT-2 padding fix
student_tokenizer.pad_token = student_tokenizer.eos_token
student_model.config.pad_token_id = student_tokenizer.eos_token_id

# =====================================================
# OPTIMIZER & TRAINING CONFIG
# =====================================================

optimizer = AdamW(student_model.parameters(), lr=1e-3)

alpha = 0.5
num_steps = 20   # demo run (mechanics verification)

# =====================================================
# TRAINING DATA
# =====================================================
# NOTE:
# STEP 5 does NOT require full Hindi corpus.
# Small samples are sufficient to verify distillation.

# ---- Option A: simple demo texts (OK academically)
train_texts = [
    "Language models predict the next token.",
    "Knowledge distillation improves students.",
    "Transformers use self attention.",
    "Neural networks learn representations."
]

# ---- Option B: OPTIONAL Hindi text hook (if you want)
# Uncomment ONLY if you have raw Hindi text lines
#
# hindi_file = Path("data/hindi_samples.txt")
# if hindi_file.exists():
#     train_texts = hindi_file.read_text(encoding="utf-8").splitlines()

# =====================================================
# TRAINING LOOP (DISTILLATION)
# =====================================================

progress = tqdm(range(num_steps))

for step in progress:
    optimizer.zero_grad()

    text = train_texts[step % len(train_texts)]

    # -------------------------
    # STUDENT FORWARD PASS
    # -------------------------
    student_inputs = student_tokenizer(
        text,
        return_tensors="pt",
        padding=True
    ).to(device)

    student_outputs = student_model(**student_inputs)
    student_logits = student_outputs.logits

    input_ids = student_inputs["input_ids"]

    # Next-token labels
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100  # ignore last token

    ce_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1),
        ignore_index=-100
    )

    # -------------------------
    # TEACHER FORWARD PASS
    # -------------------------
    with torch.no_grad():
        teacher_inputs = teacher_tokenizer(
            text,
            return_tensors="pt",
            padding=True
        ).to(device)

        input_ids_t = teacher_inputs["input_ids"].clone()

        mask_pos = input_ids_t.size(1) // 2
        input_ids_t[:, mask_pos] = teacher_tokenizer.mask_token_id

        teacher_outputs = teacher_model(
            input_ids=input_ids_t,
            attention_mask=teacher_inputs["attention_mask"]
        )

        teacher_probs = F.softmax(
            teacher_outputs.logits[:, mask_pos, :],
            dim=-1
        )

        teacher_conf = teacher_probs.max(dim=-1).values

    # -------------------------
    # STUDENT CONFIDENCE
    # -------------------------
    student_probs = F.softmax(
        student_logits[:, mask_pos, :],
        dim=-1
    )

    student_conf = student_probs.max(dim=-1).values

    distill_loss = F.mse_loss(student_conf, teacher_conf)

    # -------------------------
    # TOTAL LOSS
    # -------------------------
    total_loss = alpha * ce_loss + (1 - alpha) * distill_loss

    total_loss.backward()
    optimizer.step()

    progress.set_description(
        f"CE: {ce_loss.item():.3f} | Distill: {distill_loss.item():.5f}"
    )

# =====================================================
# STEP 5.5 — SAVE STUDENT CHECKPOINT (MANDATORY)
# =====================================================

SAVE_DIR = "student_distilled_model"
Path(SAVE_DIR).mkdir(exist_ok=True)

student_model.save_pretrained(SAVE_DIR)
student_tokenizer.save_pretrained(SAVE_DIR)

print(f"\n✅ Student model checkpoint saved at: {SAVE_DIR}")
