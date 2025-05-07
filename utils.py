#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module d'utilitaires pour le projet YOLOv5-Face
"""

import os
import subprocess
from google.colab import drive
import traceback

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
        subprocess.run(['git', 'clone', 'https://github.com/deepcam-cn/yolov5-face.git', yolo_dir], check=True)
        
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
    """Corrige le problème de np.int dans les fichiers du projet
    
    Args:
        yolo_dir (str): Répertoire de YOLOv5-Face
        
    Returns:
        bool: True si la correction a réussi, False sinon
    """
    print("\n=== Correction des erreurs de NumPy API ===\n")
    
    # Recherche récursive de tous les fichiers Python pouvant contenir np.int
    python_files = []
    for root, dirs, files in os.walk(yolo_dir):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    try:
        fixed_files = 0
        critical_files = []
        
        # Vérifier d'abord les fichiers connus pour causer des problèmes
        face_datasets_path = os.path.join(yolo_dir, 'utils', 'face_datasets.py')
        if os.path.exists(face_datasets_path):
            critical_files.append(face_datasets_path)
            print(f"✓ Fichier critique trouvé: {os.path.relpath(face_datasets_path, yolo_dir)}")
        
        # Traiter d'abord les fichiers critiques
        for file_path in critical_files:
            try:
                # Lire le contenu du fichier
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Vérifier spécifiquement pour "np.int" (et pas juste en tant que sous-chaîne)
                if "np.int" in content and not "np.int32" in content and not "np.int64" in content:
                    # Remplacer np.int par np.int32
                    modified_content = content.replace("np.int", "np.int32")
                    
                    # Sauvegarder le fichier modifié
                    with open(file_path, 'w') as f:
                        f.write(modified_content)
                    
                    print(f"✓ Correction PRIORITAIRE effectuée dans {os.path.relpath(file_path, yolo_dir)}")
                    fixed_files += 1
                    # Retirer le fichier de la liste principale pour éviter la duplication
                    python_files = [f for f in python_files if f != file_path]
            except Exception as file_error:
                print(f"✗ Erreur lors de la correction du fichier critique {file_path}: {file_error}")
        
        # Traiter ensuite tous les autres fichiers Python
        for file_path in python_files:
            try:
                # Lire le contenu du fichier
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Vérifier si le fichier contient np.int (avec des espaces ou non)
                if "np.int" in content and not "np.int32" in content and not "np.int64" in content:
                    # Remplacer np.int par np.int32
                    modified_content = content.replace("np.int", "np.int32")
                    
                    # Sauvegarder le fichier modifié
                    with open(file_path, 'w') as f:
                        f.write(modified_content)
                    
                    print(f"✓ Correction effectuée dans {os.path.relpath(file_path, yolo_dir)}")
                    fixed_files += 1
            except Exception as file_error:
                print(f"✗ Erreur lors de la vérification de {file_path}: {file_error}")
        
        if fixed_files > 0:
            print(f"\n✅ {fixed_files} fichiers corrigés avec succès")
        else:
            print("\n✅ Aucune correction nécessaire, tous les fichiers sont déjà compatibles")
        
        # Vérifier spécifiquement que face_datasets.py a été corrigé
        if os.path.exists(face_datasets_path):
            with open(face_datasets_path, 'r') as f:
                content = f.read()
            if "np.int" in content and not "np.int32" in content and not "np.int64" in content:
                print(f"\n⚠️ ATTENTION: Le fichier {os.path.relpath(face_datasets_path, yolo_dir)} contient encore np.int!")
                print("Application d'une correction directe...")
                
                # Correction forcée spécifique pour face_datasets.py
                modified_content = content.replace("np.int", "np.int32")
                with open(face_datasets_path, 'w') as f:
                    f.write(modified_content)
                print(f"✓ Correction forcée appliquée à {os.path.relpath(face_datasets_path, yolo_dir)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur lors de la correction des erreurs NumPy: {e}")
        print(traceback.format_exc())
        return False
