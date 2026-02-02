import os

def remove_empty_txt_files(folder_path):
    removed = 0

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if len(content.strip()) == 0:
                os.remove(file_path)
                removed += 1

    return removed


train_path = "data/train"
valid_path = "data/valid"

removed_train = remove_empty_txt_files(train_path)
removed_valid = remove_empty_txt_files(valid_path)

print(f"Removed {removed_train} empty files from TRAIN")
print(f"Removed {removed_valid} empty files from VALID")
