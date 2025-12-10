import json
import matplotlib.pyplot as plt

# ---------- File paths ----------
files = {
    "Original": "APIClassification_Original.txt",
    "SecureBERT": "APIClassification_Securebert.txt",
    "LLaMA": "APIClassification_Llama.txt",
}

# ---------- Categories ----------
categories = ["command_api", "database_api", "display_api", "path_api", "proxy_api"]

# ---------- Load JSON-like file ----------
def load_api_json(path):
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()
    return json.loads(txt)

# ---------- Count ----------
def count_categories(data):
    return {c: len(data.get(c, [])) for c in categories}

# ---------- Load and count ----------
results = {model: count_categories(load_api_json(path)) for model, path in files.items()}

print("\nExtracted counts:")
for k, v in results.items():
    print(k, v)

# ---------- Prepare values ----------
labels = categories
x = range(len(labels))

original_counts = [results["Original"][c] for c in labels]
secure_counts = [results["SecureBERT"][c] for c in labels]
llama_counts = [results["LLaMA"][c] for c in labels]

width = 0.25

plt.figure(figsize=(12, 6))

# bars
bars1 = plt.bar([p - width for p in x], original_counts, width=width, label="Original")
bars2 = plt.bar(x, secure_counts, width=width, label="SecureBERT")
bars3 = plt.bar([p + width for p in x], llama_counts, width=width, label="LLaMA")

# ---------- Add count labels ----------
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            str(height),
            ha='center',
            va='bottom',
            fontsize=10
        )

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

# ticks, labels
plt.xticks(x, labels, rotation=45)
plt.ylabel("API Count")
plt.title("Jellyfin API Classification Comparison")
plt.legend()
plt.tight_layout()
plt.show()
