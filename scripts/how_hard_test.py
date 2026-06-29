import pickle
import torch
import numpy as np

processed_dir = "data/processed_hybrid_bge"

train_pairs = torch.load(f"{processed_dir}/train_pairs.pt", map_location="cpu")
val_pairs = torch.load(f"{processed_dir}/val_pairs.pt", map_location="cpu")
test_pairs = torch.load(f"{processed_dir}/test_pairs.pt", map_location="cpu")

with open(f"{processed_dir}/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

train_pos = np.asarray(train_pairs["pos"])
val_pos = np.asarray(val_pairs["pos"])
test_pos = np.asarray(test_pairs["pos"])

def stat(name, pairs):
    mashups = pairs[:, 0]
    apis = pairs[:, 1]

    unique_m = np.unique(mashups)
    unique_a = np.unique(apis)

    counts = {}
    for m, a in pairs:
        counts.setdefault(int(m), 0)
        counts[int(m)] += 1

    vals = list(counts.values())

    print(f"\n{name}")
    print("pos pairs:", len(pairs))
    print("unique mashups:", len(unique_m))
    print("unique apis:", len(unique_a))
    print("avg positives per mashup:", np.mean(vals))
    print("min positives per mashup:", np.min(vals))
    print("max positives per mashup:", np.max(vals))

stat("train", train_pos)
stat("val", val_pos)
stat("test", test_pos)

train_m = set(train_pos[:, 0].tolist())
val_m = set(val_pos[:, 0].tolist())
test_m = set(test_pos[:, 0].tolist())

print("\nOverlap check")
print("train & val mashup overlap:", len(train_m & val_m))
print("train & test mashup overlap:", len(train_m & test_m))
print("val & test mashup overlap:", len(val_m & test_m))

train_api = set(train_pos[:, 1].tolist())
test_api = set(test_pos[:, 1].tolist())

print("\nAPI coverage")
print("test APIs not seen in train:", len(test_api - train_api))
print("ratio:", len(test_api - train_api) / max(len(test_api), 1))