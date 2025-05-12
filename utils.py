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
    """Corrige les problèmes d'API obsolètes de NumPy (np.int et np.float) dans les fichiers du projet
    
    Args:
        yolo_dir (str): Répertoire de YOLOv5-Face
        
    Returns:
        bool: True si la correction a réussi, False sinon
    """
    print("\n=== Correction des erreurs de NumPy API ===\n")
    
    # Liste des fichiers critiques connus pour causer des problèmes
    critical_files = [
        os.path.join(yolo_dir, 'utils', 'face_datasets.py'),
        os.path.join(yolo_dir, 'widerface_evaluate', 'box_overlaps.pyx')
    ]
    
    try:
        fixed_files = 0
        
        # 1. Corriger d'abord les fichiers critiques avec une méthode directe
        for file_path in critical_files:
            if os.path.exists(file_path):
                print(f"✓ Traitement du fichier critique: {os.path.relpath(file_path, yolo_dir)}")
                
                try:
                    # Lire le contenu du fichier
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Rechercher spécifiquement les lignes problématiques (np.int et np.float)
                    modified = False
                    modified_content = content
                    
                    # Corriger np.int
                    if 'np.int' in content:
                        # Corriger avec une expression régulière pour cibler uniquement np.int isolé
                        modified_content = re.sub(r'np\.int\b', 'np.int32', modified_content)
                        modified = True
                    
                    # Corriger np.float
                    if 'np.float' in content:
                        # Corriger avec une expression régulière pour cibler uniquement np.float isolé
                        modified_content = re.sub(r'np\.float\b', 'np.float64', modified_content)
                        modified = True
                        
                    # Si le contenu a été modifié
                    if modified_content != content:
                        # Sauvegarder le fichier modifié
                        with open(file_path, 'w') as f:
                            f.write(modified_content)
                        
                        print(f"✅ Correction effectuée dans {os.path.relpath(file_path, yolo_dir)}")
                        fixed_files += 1
                    else:
                        print(f"⚠️ Aucune modification n'a été apportée à {os.path.relpath(file_path, yolo_dir)}")
                        
                        # Rechercher et rapporter les lignes problématiques
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if 'np.int' in line:
                                print(f"    Ligne {i+1} (np.int): {line.strip()}")
                            if 'np.float' in line:
                                print(f"    Ligne {i+1} (np.float): {line.strip()}")
                            
                        # Corrections manuelles pour des cas connus
                        if "bi = np.floor(np.arange(n) / batch_size).astype(np.int)" in content:
                            modified_content = content.replace(
                                "bi = np.floor(np.arange(n) / batch_size).astype(np.int)",
                                "bi = np.floor(np.arange(n) / batch_size).astype(np.int32)"
                            )
                            with open(file_path, 'w') as f:
                                f.write(modified_content)
                            print(f"✅ Correction manuelle effectuée pour np.int")
                            fixed_files += 1
                            
                        # Correction manuelle pour np.float dans box_overlaps.pyx
                        if "DTYPE = np.float" in content:
                            modified_content = content.replace(
                                "DTYPE = np.float",
                                "DTYPE = np.float64"
                            )
                            with open(file_path, 'w') as f:
                                f.write(modified_content)
                            print(f"✅ Correction manuelle effectuée pour np.float")
                            fixed_files += 1
                except Exception as e:
                    print(f"✗ Erreur lors de la correction de {file_path}: {e}")
                    
                    # Tentative de correction avec une méthode alternative
                    try:
                        # Tenter une correction avec sed (sur les systèmes Unix)
                        subprocess.run(['sed', '-i', 's/np.int/np.int32/g', file_path], check=False)
                        subprocess.run(['sed', '-i', 's/np.float/np.float64/g', file_path], check=False)
                        print(f"✓ Tentative de correction avec sed pour {os.path.relpath(file_path, yolo_dir)}")
                    except Exception:
                        pass
            else:
                print(f"✗ Fichier critique non trouvé: {os.path.relpath(file_path, yolo_dir)}")
        
        # 2. Recherche récursive de tous les autres fichiers Python pouvant contenir np.int
        for root, dirs, files in os.walk(yolo_dir):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    
                    # Ignorer les fichiers déjà traités
                    if file_path in critical_files:
                        continue
                    
                    try:
                        # Lire le contenu du fichier
                        with open(file_path, 'r') as f:
                            content = f.read()
                        
                        # Vérifier si le fichier contient np.int ou np.float
                        has_np_int = "np.int" in content and not "np.int32" in content and not "np.int64" in content
                        has_np_float = "np.float" in content and not "np.float32" in content and not "np.float64" in content
                        
                        if has_np_int or has_np_float:
                            # Appliquer les corrections nécessaires
                            modified_content = content
                            
                            if has_np_int:
                                modified_content = re.sub(r'np\.int\b', 'np.int32', modified_content)
                                
                            if has_np_float:
                                modified_content = re.sub(r'np\.float\b', 'np.float64', modified_content)
                            
                            # Sauvegarder le fichier modifié
                            with open(file_path, 'w') as f:
                                f.write(modified_content)
                            
                            print(f"✓ Correction effectuée dans {os.path.relpath(file_path, yolo_dir)}")
                            fixed_files += 1
                    except Exception as file_error:
                        print(f"✗ Erreur lors de la vérification de {file_path}: {file_error}")
        
        # 3. Vérification finale des fichiers critiques
        for file_path in critical_files:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Vérifier spécifiquement si des problèmes persistent
                has_np_int = "np.int" in content and not "np.int32" in content and not "np.int64" in content
                has_np_float = "np.float" in content and not "np.float32" in content and not "np.float64" in content
                
                if has_np_int or has_np_float:
                    if has_np_int:
                        print(f"\n❌ AVERTISSEMENT: Le fichier {os.path.relpath(file_path, yolo_dir)} contient encore np.int!")
                    if has_np_float:
                        print(f"\n❌ AVERTISSEMENT: Le fichier {os.path.relpath(file_path, yolo_dir)} contient encore np.float!")
                    print("Tentative de correction directe avec un remplacement forcé:")
                    
                    # Utiliser une méthode plus radicale avec un motif très spécifique
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                    
                    # Trouver et corriger les lignes spécifiques
                    for i, line in enumerate(lines):
                        if has_np_int and "np.int" in line:
                            if ".astype(np.int)" in line:
                                lines[i] = line.replace(".astype(np.int)", ".astype(np.int32)")
                                print(f"✅ Ligne {i+1} corrigée (np.int): {lines[i].strip()}")
                            elif "DTYPE = np.int" in line or "dtype=np.int" in line:
                                lines[i] = line.replace("np.int", "np.int32")
                                print(f"✅ Ligne {i+1} corrigée (np.int): {lines[i].strip()}")
                                
                        if has_np_float and "np.float" in line:
                            if ".astype(np.float)" in line:
                                lines[i] = line.replace(".astype(np.float)", ".astype(np.float64)")
                                print(f"✅ Ligne {i+1} corrigée (np.float): {lines[i].strip()}")
                            elif "DTYPE = np.float" in line or "dtype=np.float" in line:
                                lines[i] = line.replace("np.float", "np.float64")
                                print(f"✅ Ligne {i+1} corrigée (np.float): {lines[i].strip()}")
                    
                    # Sauvegarder le fichier modifié
                    with open(file_path, 'w') as f:
                        f.writelines(lines)
                    
                    # Vérifier que la correction a bien été appliquée
                    with open(file_path, 'r') as f:
                        new_content = f.read()
                    
                    still_has_np_int = "np.int" in new_content and "np.int32" not in new_content and "np.int64" not in new_content
                    still_has_np_float = "np.float" in new_content and "np.float32" not in new_content and "np.float64" not in new_content
                    
                    if still_has_np_int:
                        print(f"❌ La correction de np.int a échoué. Veuillez corriger manuellement le fichier {os.path.relpath(file_path, yolo_dir)}")
                        
                    if still_has_np_float:
                        print(f"❌ La correction de np.float a échoué. Veuillez corriger manuellement le fichier {os.path.relpath(file_path, yolo_dir)}")
        
        if fixed_files > 0:
            print(f"\n✅ {fixed_files} fichiers corrigés avec succès")
        else:
            print("\n✅ Aucune correction nécessaire, tous les fichiers sont déjà compatibles")
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur lors de la correction des erreurs NumPy: {e}")
        print(traceback.format_exc())
        return False
