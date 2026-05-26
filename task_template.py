
## The code below is to fetch the model from hugging face 

from huggingface_hub import login, snapshot_download

hf_token = "Your Hugging face token"  

login(token=hf_token)

repo_dir = snapshot_download(
    repo_id="Repo id",  ## The repo were the model is present 
    repo_type="model",
    token=hf_token,
    local_dir="tm126_task2"
)

print(repo_dir)

## Main Task template code

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader, Subset
from safetensors.torch import load_file
import pandas as pd
import numpy as np


def make_model():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 100)
    return model


def load_model(path):
    model = make_model()
    weights = load_file(path, device="cpu")
    model.load_state_dict(weights, strict=True)
    model.eval()
    return model


def normalize(x):
    x = np.array(x, dtype=np.float64)
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


def weight_cosine(target, suspect):
    target_weights = []
    suspect_weights = []

    for t, s in zip(target.state_dict().values(), suspect.state_dict().values()):
        if torch.is_tensor(t) and t.dtype.is_floating_point:
            target_weights.append(t.detach().cpu().float().flatten())
            suspect_weights.append(s.detach().cpu().float().flatten())

    target_weights = torch.cat(target_weights)
    suspect_weights = torch.cat(suspect_weights)

    return F.cosine_similarity(target_weights, suspect_weights, dim=0).item()


@torch.no_grad()
def knockoff_scores(target, suspect, loader, device):
    target.to(device).eval()
    suspect.to(device).eval()

    agreements = []
    logit_sims = []
    prob_sims = []

    for images, _ in loader:
        images = images.to(device)

        target_logits = target(images)
        suspect_logits = suspect(images)

        agreement = (
            target_logits.argmax(1) == suspect_logits.argmax(1)
        ).float().mean().item()

        logit_sim = F.cosine_similarity(
            target_logits,
            suspect_logits,
            dim=1
        ).mean().item()

        target_prob = F.softmax(target_logits, dim=1)
        suspect_prob = F.softmax(suspect_logits, dim=1)

        prob_sim = F.cosine_similarity(
            target_prob,
            suspect_prob,
            dim=1
        ).mean().item()

        agreements.append(agreement)
        logit_sims.append(logit_sim)
        prob_sims.append(prob_sim)

    return np.mean(agreements), np.mean(logit_sims), np.mean(prob_sims)


@torch.no_grad()
def feature_similarity(target, suspect, loader, device):
    target.to(device).eval()
    suspect.to(device).eval()

    target_feature = []
    suspect_feature = []

    def save_target_feature(module, input, output):
        target_feature.clear()
        target_feature.append(output.detach())

    def save_suspect_feature(module, input, output):
        suspect_feature.clear()
        suspect_feature.append(output.detach())

    hook1 = target.layer4.register_forward_hook(save_target_feature)
    hook2 = suspect.layer4.register_forward_hook(save_suspect_feature)

    scores = []

    for images, _ in loader:
        images = images.to(device)

        _ = target(images)
        _ = suspect(images)

        f1 = target_feature[0].flatten(1)
        f2 = suspect_feature[0].flatten(1)

        sim = F.cosine_similarity(f1, f2, dim=1).mean().item()
        scores.append(sim)

    hook1.remove()
    hook2.remove()

    return np.mean(scores)


device = "cuda" if torch.cuda.is_available() else "cpu"

target = load_model("tm126_task2/target_model/weights.safetensors")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5071, 0.4867, 0.4408),
        (0.2675, 0.2565, 0.2761)
    )
])

dataset = datasets.CIFAR100(
    root="/kaggle/working/data",
    train=False,
    download=True,
    transform=transform
)

dataset = Subset(dataset, range(3000))

loader = DataLoader(
    dataset,
    batch_size=256,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

rows = []

for i in range(360):
    path = f"tm126_task2/suspect_models/suspect_{i:03d}.safetensors"
    suspect = load_model(path)

    w = weight_cosine(target, suspect)
    agreement, logit, prob = knockoff_scores(target, suspect, loader, device)
    feature = feature_similarity(target, suspect, loader, device)

    rows.append({
        "id": i,
        "weight": w,
        "agreement": agreement,
        "logit": logit,
        "prob": prob,
        "feature": feature
    })

    print(i, w, agreement, logit, prob, feature)

    suspect.cpu()
    del suspect

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


df = pd.DataFrame(rows)

df["weight"] = normalize(df["weight"])
df["agreement"] = normalize(df["agreement"])
df["logit"] = normalize(df["logit"])
df["prob"] = normalize(df["prob"])
df["feature"] = normalize(df["feature"])

df["score"] = (
    0.35 * df["weight"] +
    0.20 * df["agreement"] +
    0.20 * df["logit"] +
    0.05 * df["prob"] +
    0.20 * df["feature"]
)

submission = df[["id", "score"]]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Saved submission.csv")


