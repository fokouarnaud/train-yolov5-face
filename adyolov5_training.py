#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script spécifique pour l'entraînement et l'utilisation d'ADYOLOv5-Face dans Google Colab
Ce script facilite l'utilisation du modèle ADYOLOv5-Face avec le mécanisme Gather-and-Distribute (GD)
et la tête de détection supplémentaire pour les petits visages
"""

import os
import argparse
import subprocess
import time
from pathlib import Path

# Monter Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Créer le dossier de travail
os.makedirs("/content", exist_ok=True)

# Définir les chemins
SCRIPTS_PATH = "/content/drive/MyDrive/yolov5_face_scripts"
RESULTS_PATH = "/content/drive/MyDrive/YOLOv5_Face_Results/ADYOLOv5_Face"
os.makedirs(RESULTS_PATH, exist_ok=True)

# Copier les scripts nécessaires
script_files = [
    'main.py',
    'data_preparation.py',
    'model_training.py',
    'model_evaluation.py',
    'utils.py',
    'colab_setup.py',
    'config.py'
]

print("=== Copie des scripts depuis Google Drive ===")
for script in script_files:
    src_path = f"{SCRIPTS_PATH}/{script}"
    dst_path = f"/content/{script}"
    subprocess.run(["cp", src_path, dst_path], check=True)
    print(f"✓ {script} copié")

# Installer les dépendances
print("\n=== Installation des dépendances ===")
subprocess.run(["pip", "install", "numpy==1.26.4", "scipy==1.13.1", "gensim==4.3.3", "--no-deps"], check=True)
subprocess.run(["pip", "install", "torch>=2.0.0", "torchvision>=0.15.0"], check=True)
subprocess.run(["pip", "install", "opencv-python"], check=True)
subprocess.run(["pip", "install", "--upgrade", "nvidia-cudnn-cu11", "nvidia-cublas-cu11"], check=True)
subprocess.run(["pip", "install", "werkzeug"], check=True)

# Définir les paramètres d'entraînement
def parse_args():
    parser = argparse.ArgumentParser(description='ADYOLOv5-Face Training')
    parser.add_argument('--batch-size', type=int, default=32, 
                        help='Taille du batch pour l\'entraînement')
    parser.add_argument('--epochs', type=int, default=300, 
                        help='Nombre d\'epochs d\'entraînement')
    parser.add_argument('--img-size', type=int, default=640, 
                        help='Taille d\'image pour l\'entraînement')
    parser.add_argument('--model-type', type=str, choices=['standard', 'simple'], default='simple',
                        help='Type de modèle (standard: avec GatherLayer/DistributeLayer, simple: implémentation alternative avec Concat)')
    parser.add_argument('--skip-train', action='store_true',
                        help='Ignorer l\'étape d\'entraînement')
    parser.add_argument('--skip-evaluation', action='store_true',
                        help='Ignorer l\'étape d\'évaluation')
    parser.add_argument('--skip-export', action='store_true',
                        help='Ignorer l\'étape d\'exportation')
    return parser.parse_args()

args = parse_args()

# Configuration de l'environnement
print("\n=== Configuration de l'environnement ===\n")
print(f"Utilisation du modèle ADYOLOv5-Face {'SIMPLE (avec Concat)' if args.model_type == 'simple' else 'STANDARD (avec GatherLayer/DistributeLayer)'}")

os.chdir("/content")

# Copier le fichier YAML spécifique selon le modèle choisi
model_yaml = "adyolov5s_simple.yaml" if args.model_type == "simple" else "adyolov5s.yaml"

# Modifier la configuration pour utiliser le bon fichier YAML
subprocess.run(
    ["python", "colab_setup.py", "--model-size", "ad", "--model-yaml", model_yaml], 
    check=True
)

# Lancer l'entraînement avec ADYOLOv5-Face
print("\n=== Lancement de l'entraînement ADYOLOv5-Face ===")
command = [
    "python", "main.py", 
    "--model-size", "ad",
    "--batch-size", str(args.batch_size),
    "--epochs", str(args.epochs),
    "--img-size", str(args.img_size),
    "--model-type", args.model_type
]

if args.skip_train:
    command.append("--skip-train")
if args.skip_evaluation:
    command.append("--skip-evaluation")
if args.skip_export:
    command.append("--skip-export")

subprocess.run(command, check=True)

# Copier les résultats vers Google Drive
print("\n=== Copie des résultats vers Google Drive ===")
subprocess.run(["cp", "-r", "/content/yolov5-face/runs/train/face_detection_transfer", RESULTS_PATH], check=True)

# Visualiser les résultats
print("\n=== Visualisation des résultats ===")
print("Chargement de TensorBoard...")
subprocess.run(["load_ext", "tensorboard"], shell=True)
subprocess.run(["tensorboard", "--logdir", "/content/yolov5-face/runs/train/face_detection_transfer"], shell=True)

print("\n=== Entraînement ADYOLOv5-Face terminé! ===")
print(f"Les résultats sont disponibles à: {RESULTS_PATH}")
print("""
Performance attendue sur WiderFace:
- Easy:   94,80% (vs 94,33% pour YOLOv5s-Face)
- Medium: 93,77% (vs 92,61% pour YOLOv5s-Face)
- Hard:   84,37% (vs 83,15% pour YOLOv5s-Face)
""")
