#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module d'utilitaires pour le projet YOLOv5-Face
"""

import os
import subprocess
from google.colab import drive
import traceback
import re

# Importer la configuration centralisée
from config import REPO_URL, DEPENDENCIES, DEFAULT_PATHS, INFO_MESSAGES

def setup_environment(yolo_dir):
    """Configure l'environnement pour YOLOv5-Face
    
    Args:
        yolo_dir (str): Répertoire de YOLOv5-Face
        
    Returns:
        bool: True si la configuration a réussi, False sinon
    """
    print("=== Configuration de l'environnement ===")
    
    try:
        # Monter Google Drive si ce n'est pas déjà fait
        if not os.path.exists('/content/drive'):
            print("Montage de Google Drive...")
            drive.mount('/content/drive')
        
        # Supprimer le répertoire yolov5-face s'il existe déjà
        if os.path.exists(yolo_dir):
            print(f"Suppression du répertoire existant: {yolo_dir}")
            subprocess.run(['rm', '-rf', yolo_dir], check=True)
        
        # Cloner le dépôt YOLOv5-Face
        print("Clonage du dépôt YOLOv5-Face...")
        print(f"Utilisation du dépôt forké: {REPO_URL}")
        print("Ce dépôt inclut les corrections pour PyTorch 2.6+ et NumPy 1.26+,")
        print("ainsi que le support pour les modèles ultra-légers (n-0.5, n) avec ShuffleNetV2")
        subprocess.run(['git', 'clone', REPO_URL, yolo_dir], check=True)
        
        # Aller dans le répertoire YOLOv5-Face
        os.chdir(yolo_dir)
        
        # Créer ou mettre à jour le fichier requirements.txt
        create_requirements_file(yolo_dir)
        
        # Installer les dépendances avec une priorité pour les versions compatibles
        print("Installation des dépendances...")
        # D'abord installer des versions spécifiques des packages critiques
        subprocess.run(['pip', 'install', 'numpy==1.26.4', 'scipy==1.13.1', 'gensim==4.3.3', '--no-deps'], check=True)
        
        # Puis installer les autres dépendances du requirements.txt
        subprocess.run(['pip', 'install', '-r', 'requirements.txt', '--no-deps'], check=True)
        
        # S'assurer que torch et torchvision sont installés
        subprocess.run(['pip', 'install', 'torch>=2.0.0', 'torchvision>=0.15.0'], check=True)
        
        print("✓ Configuration de l'environnement terminée")
        return True
        
    except Exception as e:
        print(f"✗ Erreur lors de la configuration de l'environnement: {e}")
        print(traceback.format_exc())
        return False

def create_requirements_file(yolo_dir):
    """Crée un fichier requirements.txt avec les dépendances compatibles"""
    requirements = """torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.5.0
tqdm>=4.64.0
PyYAML>=6.0
seaborn>=0.12.0
scipy<=1.13.1
thop>=0.1.1
requests>=2.27.0
Cython>=0.29.0
onnx>=1.12.0
onnxruntime>=1.10.0
tensorboard>=2.8.0
gensim==4.3.3"""  # Spécifier la version exacte de gensim

    with open(f'{yolo_dir}/requirements.txt', 'w') as f:
        f.write(requirements)
    
    print("✓ Fichier requirements.txt créé")

def fix_numpy_issue(yolo_dir):
    """Affiche des instructions pour corriger les problèmes d'API obsolètes de NumPy
    
    Args:
        yolo_dir (str): Répertoire de YOLOv5-Face
        
    Returns:
        bool: True toujours (fonction informative uniquement)
    """
    print("\n=== Instructions pour corriger les erreurs de NumPy API ===\n")
    print(INFO_MESSAGES["numpy_fix"])
    
    # Liste des fichiers critiques connus pour causer des problèmes
    critical_files = [
        os.path.join(yolo_dir, 'utils', 'face_datasets.py'),
        os.path.join(yolo_dir, 'widerface_evaluate', 'box_overlaps.pyx')
    ]
    
    print("\nLes corrections ont été intégrées dans le dépôt forké, mais si vous utilisez le dépôt original,")
    print("vous devrez corriger ces fichiers manuellement pour NumPy 1.26+ et Python 3.11:")
    print("\n1. box_overlaps.pyx (dans widerface_evaluate/)")
    print("   - Remplacer np.int par np.int64")
    print("   - Remplacer np.int_t par np.int64_t")
    print("   - Remplacer np.float par np.float64")
    print("\n2. utils/face_datasets.py")
    print("   - Remplacer .astype(np.int) par .astype(np.int32)")
    print("   - Remplacer np.float par np.float64")
    
    return True
