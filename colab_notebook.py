#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script complet pour le notebook Google Colab
Incluant toutes les étapes nécessaires pour l'entraînement de YOLOv5-Face
"""

# Monter Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Afficher la version de Python et les informations GPU
!python --version
!nvidia-smi

# Créer un répertoire pour les scripts Python
!mkdir -p /content/yolov5_scripts

# Copier les scripts Python depuis votre dossier Drive (adaptez le chemin si nécessaire)
!cp /content/drive/MyDrive/yolov5_face_scripts/*.py /content/

# Créer un fichier __init__.py pour que les modules soient reconnus
!touch /content/__init__.py

# Ajouter le répertoire courant au path Python
import sys
sys.path.insert(0, '/content')

# Vérifier les fichiers présents
!ls -la /content/*.py

# Installer les dépendances compatibles
!pip install numpy==1.26.4 scipy==1.13.1 gensim==4.3.3 --no-deps

# Cloner le dépôt YOLOv5-Face
!git clone https://github.com/deepcam-cn/yolov5-face.git /content/yolov5-face

# Créer le répertoire des poids
!mkdir -p /content/yolov5-face/weights

# Télécharger les poids de la version 5.0
!wget -O /content/yolov5-face/weights/yolov5s.pt https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt

# Modifier le script train.py pour être compatible avec PyTorch 2.6+
import os

train_path = '/content/yolov5-face/train.py'
if os.path.exists(train_path):
    # Lire le contenu du fichier
    with open(train_path, 'r') as f:
        content = f.read()
    
    # Remplacer la ligne torch.load
    modified_content = content.replace(
        'torch.load(weights, map_location=device)',
        'torch.load(weights, map_location=device, weights_only=False)'
    )
    
    # Sauvegarder le fichier modifié
    with open(train_path, 'w') as f:
        f.write(modified_content)
    print(f"✓ Fichier {train_path} modifié avec succès")

# Exécuter le script principal pour l'entraînement
%cd /content
!python main.py --yolo-version 5.0 --model-size s --img-size 640 --batch-size 32

# Décommenter pour visualiser les résultats avec TensorBoard
# %load_ext tensorboard
# %tensorboard --logdir /content/yolov5-face/runs/train/face_detection_transfer

# Décommenter pour sauvegarder les résultats sur Google Drive
# !mkdir -p /content/drive/MyDrive/YOLOv5_Face_Results
# !cp -r /content/yolov5-face/runs/train/face_detection_transfer /content/drive/MyDrive/YOLOv5_Face_Results/
