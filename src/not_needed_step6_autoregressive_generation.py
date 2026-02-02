import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL_NAME = "distilgpt2"     # student architecture
CHECKPOINT_PATH = "./student_checkpoint"  
# ↑ change if your trained model is saved elsewhere

MAX_NEW_TOKENS = 40
TEMPERATURE = 1.0

# --------------------------------------------------
# LOAD TOKENIZER & MODEL
# --------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# DistilGPT-2 has no PAD token → use EOS as PAD
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    CHECKPOINT_PATH if CHECKPOINT_PATH else MODEL_NAME
)

model.eval()

# --------------------------------------------------
# AUTOREGRESSIVE GENERATION FUNCTION
# --------------------------------------------------

def generate_autoregressive_text(
    model,
    tokenizer,
    prompt,
    max_new_tokens=40,
    temperature=1.0
):
    """
    Generates text by discrete next-token classification
    using autoregressive decoding.
    """

    # Encode prompt → discrete token IDs
    input_ids = tokenizer(
        prompt,
        return_tensors="pt"
    ).input_ids

    for step in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits  # (1, seq_len, vocab_size)

        # Select logits of last generated token
        next_token_logits = logits[:, -1, :] / temperature

        # Convert logits → probability distribution
        probs = torch.softmax(next_token_logits, dim=-1)

        # Sample ONE discrete token (classification)
        next_token_id = torch.multinomial(probs, num_samples=1)

        # (Optional) Print token-by-token generation
        decoded_token = tokenizer.decode(next_token_id[0])
        print(f"Step {step+1}: {decoded_token}")

        # Append token to input
        input_ids = torch.cat([input_ids, next_token_id], dim=1)

    # Decode full sequence
    generated_text = tokenizer.decode(
        input_ids[0],
        skip_special_tokens=True
    )

    return generated_text

# --------------------------------------------------
# MAIN DEMO
# --------------------------------------------------

if __name__ == "__main__":

    prompt = "Language models generate text"

    print("\nPROMPT:")
    print(prompt)
    print("\nGENERATING...\n")

    output_text = generate_autoregressive_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE
    )

    print("\nFINAL GENERATED TEXT:\n")
    print(output_text)
