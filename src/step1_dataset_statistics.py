import os

def compute_dataset_statistics(folder_path):
    file_lengths = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                file_lengths.append(len(text))

    num_docs = len(file_lengths)
    total_chars = sum(file_lengths)
    avg_chars = total_chars / num_docs if num_docs > 0 else 0
    min_chars = min(file_lengths) if file_lengths else 0
    max_chars = max(file_lengths) if file_lengths else 0

    return {
        "documents": num_docs,
        "total_characters": total_chars,
        "average_characters": avg_chars,
        "min_characters": min_chars,
        "max_characters": max_chars
    }


# ---- Paths (update if needed) ----
train_path = "data/train"
valid_path = "data/valid"

# ---- Compute statistics ----
train_stats = compute_dataset_statistics(train_path)
valid_stats = compute_dataset_statistics(valid_path)

# ---- Print results ----
print("📘 TRAIN DATASET STATISTICS")
for k, v in train_stats.items():
    print(f"{k}: {v}")

print("\n📗 VALIDATION DATASET STATISTICS")
for k, v in valid_stats.items():
    print(f"{k}: {v}")
