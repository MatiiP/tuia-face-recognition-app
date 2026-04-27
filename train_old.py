"""
train.py — Script de entrenamiento del modelo de reconocimiento facial.
Fine-tunes a ResNet-18 (pretrained on ImageNet) to produce 512-dim face embeddings.

Reglas del dataset:
  - valentino: excluido totalmente (para pruebas como unknown)
  - ambrogi: se toman TODAS las imágenes (prueba de desbalance)
  - resto: se toman hasta 10 imágenes por clase

Al finalizar, exporta:
  - models/face_detection.pth  (nn.Sequential: backbone + embedding_head)
  - output/training_metrics.json  (loss y accuracy por época)
"""

import os
import glob
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from PIL import Image
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# ──────────────────────────── Configuración ────────────────────────────
DATASET_DIR = "Dataset"
MODEL_SAVE_PATH = "models/face_detection.pth"
METRICS_SAVE_PATH = "output/training_metrics.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_SIZE = 512
FACE_SIZE = 112
EPOCHS = 50
BATCH_SIZE = 8
LR = 1e-4

# ──────────────────────────── 1. Detector de rostros ────────────────────────────
print(f"Iniciando entrenamiento en {DEVICE}...")
analyzer = FaceAnalysis(name="buffalo_s", allowed_modules=["detection"])
analyzer.prepare(ctx_id=0 if DEVICE == "cuda" else -1, det_size=(640, 640))

# ──────────────────────────── 2. Recopilar dataset ────────────────────────────
classes = []
image_paths = []
labels = []

for cls_name in sorted(os.listdir(DATASET_DIR)):
    cls_path = os.path.join(DATASET_DIR, cls_name)
    if not os.path.isdir(cls_path):
        continue
    if cls_name.lower() == "valentino":
        print(f"Saltando a {cls_name} (excluido del entrenamiento).")
        continue
    classes.append(cls_name)

num_classes = len(classes)
print(f"Clases válidas encontradas: {num_classes}")

for label_idx, cls_name in enumerate(classes):
    cls_path = os.path.join(DATASET_DIR, cls_name)
    files = sorted(glob.glob(os.path.join(cls_path, "*.*")))

    if cls_name.lower() == "ambrogi":
        selected = files
        print(f"  {cls_name}: {len(selected)} imágenes (todas, prueba de desbalance)")
    else:
        selected = files[:10]
        print(f"  {cls_name}: {len(selected)} imágenes (cap 10)")

    for f in selected:
        image_paths.append(f)
        labels.append(label_idx)

# ──────────────────────────── 3. Dataset con detección + alineación ────────────────────────────
class FaceDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels[idx]

        img_bgr = cv2.imread(path)
        if img_bgr is None:
            return torch.zeros((3, FACE_SIZE, FACE_SIZE)), label

        faces = analyzer.get(img_bgr)
        if len(faces) > 0:
            faces = sorted(
                faces,
                key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
                reverse=True,
            )
            aligned_bgr = face_align.norm_crop(
                img_bgr, landmark=faces[0].kps, image_size=FACE_SIZE
            )
        else:
            aligned_bgr = cv2.resize(img_bgr, (FACE_SIZE, FACE_SIZE))

        aligned_rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(Image.fromarray(aligned_rgb))
        return tensor, label


train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = FaceDataset(image_paths, labels, train_transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ──────────────────────────── 4. Modelo ────────────────────────────
class FaceEmbeddingModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = timm.create_model("resnet18", pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.embedding_head = nn.Linear(in_features, EMBEDDING_SIZE)
        self.classifier = nn.Linear(EMBEDDING_SIZE, num_classes)

    def forward(self, x, return_embedding=False):
        features = self.backbone(x)
        embedding = self.embedding_head(features)
        if return_embedding:
            return embedding
        return self.classifier(nn.functional.relu(embedding))


model = FaceEmbeddingModel(num_classes).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ──────────────────────────── 5. Loop de entrenamiento ────────────────────────────
if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    metrics_history = []

    print(f"\nIniciando loop de entrenamiento ({EPOCHS} épocas)...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / len(loader)
        epoch_acc = 100.0 * correct / total
        metrics_history.append({
            "epoch": epoch + 1,
            "loss": round(epoch_loss, 4),
            "accuracy": round(epoch_acc, 2),
        })
        print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {epoch_loss:.4f} — Acc: {epoch_acc:.2f}%")

    # ──────────────────────────── 6. Exportar modelo ────────────────────────────
    final_model = nn.Sequential(model.backbone, model.embedding_head)
    final_model.eval()
    torch.save(final_model, MODEL_SAVE_PATH)
    print(f"\nModelo guardado en {MODEL_SAVE_PATH}")

    # ──────────────────────────── 7. Guardar métricas ────────────────────────────
    metrics_payload = {
        "config": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "embedding_size": EMBEDDING_SIZE,
            "face_size": FACE_SIZE,
            "device": DEVICE,
            "num_classes": num_classes,
            "class_names": classes,
            "total_images": len(image_paths),
        },
        "history": metrics_history,
        "final_accuracy": metrics_history[-1]["accuracy"],
        "final_loss": metrics_history[-1]["loss"],
    }
    with open(METRICS_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)
    print(f"Métricas guardadas en {METRICS_SAVE_PATH}")
    print("¡Entrenamiento completado!")
