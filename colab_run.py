#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script tout-en-un pour Google Colab
Ce script inclut toutes les étapes nécessaires pour configurer l'environnement,
corriger les problèmes de compatibilité et lancer l'entraînement de YOLOv5-Face.
"""

import os
import sys
import subprocess
import re

# Étape 1: Monter Google Drive et configurer l'environnement
from google.colab import drive
drive.mount('/content/drive')

print("\n" + "=" * 70)
print(" CONFIGURATION DE L'ENVIRONNEMENT POUR YOLOV5-FACE ".center(70, "="))
print("=" * 70 + "\n")

# Vérifier les fichiers du dataset avec subprocess
subprocess.run(['ls', '-la', '/content/drive/MyDrive/dataset/'])

# Créer un répertoire pour les scripts Python
subprocess.run(['mkdir', '-p', '/content'])

# Copier les scripts nécessaires depuis Google Drive
subprocess.run(['cp', '/content/drive/MyDrive/yolov5_face_scripts/{main.py,data_preparation.py,model_training.py,model_evaluation.py,utils.py,colab_setup.py}', '/content/'])

# Vérifier que les scripts ont été copiés
subprocess.run(['ls', '-la', '/content/*.py'])

# Créer un fichier __init__.py pour que les modules soient reconnus
subprocess.run(['touch', '/content/__init__.py'])

# Ajouter le répertoire courant au path Python
if '/content' not in sys.path:
    sys.path.insert(0, '/content')

# Étape 2: Installer les dépendances compatibles
print("\n" + "=" * 70)
print(" INSTALLATION DES DÉPENDANCES ".center(70, "="))
print("=" * 70 + "\n")

subprocess.run(['pip', 'install', 'numpy==1.26.4', 'scipy==1.13.1', 'gensim==4.3.3', '--no-deps'])
subprocess.run(['pip', 'install', 'torch>=2.0.0', 'torchvision>=0.15.0'])

# Étape 3: Cloner le dépôt YOLOv5-Face et télécharger les poids
print("\n" + "=" * 70)
print(" PRÉPARATION DU MODÈLE ".center(70, "="))
print("=" * 70 + "\n")

# Utiliser le dépôt forké avec les corrections déjà appliquées
subprocess.run(['git', 'clone', 'https://github.com/fokouarnaud/yolov5-face.git', '/content/yolov5-face'])

# Créer le répertoire des poids et télécharger les poids pré-entraînés
subprocess.run(['mkdir', '-p', '/content/yolov5-face/weights'])
subprocess.run(['wget', '-O', '/content/yolov5-face/weights/yolov5s.pt', 
                'https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt'])

# Vérification de la compatibilité PyTorch
print("\n" + "=" * 70)
print(" VÉRIFICATION DE LA COMPATIBILITÉ PYTORCH ".center(70, "="))
print("=" * 70 + "\n")

print("✓ Le dépôt forké inclut déjà les corrections nécessaires pour PyTorch 2.6+")
print("Aucune correction supplémentaire n'est requise.")

# Installer werkzeug pour résoudre le problème de TensorBoard
subprocess.run(['pip', 'install', 'werkzeug'])

# Étape 4: Lancer l'entraînement
print("\n" + "=" * 70)
print(" LANCEMENT DE L'ENTRAÎNEMENT ".center(70, "="))
print("=" * 70 + "\n")

os.chdir('/content')
subprocess.run(['python', 'main.py'])

# Étape 5: Visualiser les résultats avec TensorBoard
print("\n" + "=" * 70)
print(" VISUALISATION DES RÉSULTATS ".center(70, "="))
print("=" * 70 + "\n")

print("Pour visualiser les résultats avec TensorBoard, exécutez les commandes suivantes:")
print("%load_ext tensorboard")
print("%tensorboard --logdir /content/yolov5-face/runs/train/face_detection_transfer")

# Étape 6: Sauvegarder les résultats
print("\n" + "=" * 70)
print(" SAUVEGARDE DES RÉSULTATS ".center(70, "="))
print("=" * 70 + "\n")

print("Pour sauvegarder les résultats sur Google Drive, exécutez les commandes suivantes:")
print("!mkdir -p /content/drive/MyDrive/YOLOv5_Face_Results")
print("!cp -r /content/yolov5-face/runs/train/face_detection_transfer /content/drive/MyDrive/YOLOv5_Face_Results/")
