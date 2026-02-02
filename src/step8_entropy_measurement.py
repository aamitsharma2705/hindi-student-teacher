import torch
import sentencepiece as spm
from transformers import GPT2LMHeadModel
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sp = spm.SentencePieceProcessor()
sp.load("tokenizers/hindi_bpe.model")

def avg_entropy(model, prompt, steps=30):
    model.eval()
    input_ids = torch.tensor(
        [sp.encode(prompt, out_type=int)],
        device=device
    )

    entropies = []

    for _ in range(steps):
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits
        probs = F.softmax(logits[:, -1, :], dim=-1)

        entropy = -(probs * probs.log()).sum().item()
        entropies.append(entropy)

        next_token = torch.multinomial(probs, 1)
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return sum(entropies) / len(entropies)

untrained = GPT2LMHeadModel.from_pretrained("student_hindi_untrained").to(device)
trained = GPT2LMHeadModel.from_pretrained("student_hindi_model").to(device)

prompt = "आप कैसे हैं"

print("Untrained entropy:", avg_entropy(untrained, prompt))
print("Trained entropy:", avg_entropy(trained, prompt))
