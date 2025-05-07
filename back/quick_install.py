#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script d'installation rapide pour Google Colab
Ce script effectue toutes les étapes nécessaires pour configurer l'environnement
et appliquer les corrections de compatibilité avant l'entraînement.
"""

import os
import sys
import subprocess
from google.colab import drive

def setup_colab_environment():
    """
    Configure l'environnement Colab pour l'entraînement YOLOv5-Face
    """
    print("=" * 80)
    print("INSTALLATION RAPIDE POUR YOLOV5-FACE")
    print("=" * 80)
    
    # 1. Monter Google Drive
    if not os.path.exists('/content/drive'):
        print("\n=== Montage de Google Drive ===")
        drive.mount('/content/drive')
    
    # 2. Vérifier les fichiers du dataset
    dataset_path = '/content/drive/MyDrive/dataset'
    required_files = [
        'WIDER_train.zip',
        'WIDER_val.zip',
        'WIDER_test.zip',
        'retinaface_gt.zip'
    ]
    
    print("\n=== Vérification des fichiers du dataset ===")
    if os.path.exists(dataset_path):
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(dataset_path, f))]
        if missing_files:
            print(f"⚠️ Fichiers manquants: {', '.join(missing_files)}")
            print(f"Veuillez télécharger ces fichiers dans {dataset_path}")
        else:
            print("✓ Tous les fichiers du dataset sont présents!")
    else:
        print(f"⚠️ Répertoire du dataset non trouvé: {dataset_path}")
        print("Création du répertoire...")
        os.makedirs(dataset_path, exist_ok=True)
        print(f"⚠️ Veuillez télécharger les fichiers du dataset dans {dataset_path}")
    
    # 3. Installer les dépendances critiques
    print("\n=== Installation des dépendances compatibles ===")
    subprocess.run(['pip', 'install', 'numpy==1.26.4', 'scipy==1.13.1', 'gensim==4.3.3', '--no-deps'], check=True)
    subprocess.run(['pip', 'install', 'torch>=2.0.0', 'torchvision>=0.15.0'], check=True)
    
    # 4. Cloner le dépôt YOLOv5-Face
    yolo_dir = '/content/yolov5-face'
    print("\n=== Configuration du dépôt YOLOv5-Face ===")
    if os.path.exists(yolo_dir):
        print(f"Suppression du dépôt existant: {yolo_dir}")
        subprocess.run(['rm', '-rf', yolo_dir], check=True)
    
    print("Clonage du dépôt...")
    subprocess.run(['git', 'clone', 'https://github.com/deepcam-cn/yolov5-face.git', yolo_dir], check=True)
    
    # 5. Télécharger les poids pré-entraînés
    weights_dir = os.path.join(yolo_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    
    model_size = 's'  # Par défaut, utiliser le modèle small
    print(f"\n=== Téléchargement des poids YOLOv5{model_size} ===")
    weights_url = f'https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5{model_size}.pt'
    weights_path = os.path.join(weights_dir, f'yolov5{model_size}.pt')
    
    subprocess.run(['wget', weights_url, '-O', weights_path], check=True)
    print(f"✓ Poids téléchargés: {weights_path}")
    
    # 6. Appliquer les corrections
    print("\n=== Application des corrections de compatibilité ===")
    
    # Correction pour NumPy
    numpy_fix_success = False
    try:
        from utils import fix_numpy_issue
        numpy_fix_success = fix_numpy_issue(yolo_dir)
    except ImportError:
        print("⚠️ Module utils non trouvé. La correction NumPy sera ignorée.")
    
    if not numpy_fix_success:
        print("Application manuelle de la correction NumPy...")
        python_files = []
        for root, dirs, files in os.walk(yolo_dir):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        fixed_files = 0
        for file_path in python_files:
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                if "np.int" in content and not "np.int32" in content and not "np.int64" in content:
                    modified_content = content.replace("np.int", "np.int32")
                    with open(file_path, 'w') as f:
                        f.write(modified_content)
                    fixed_files += 1
            except Exception as e:
                print(f"Erreur lors de la vérification de {file_path}: {e}")
        
        print(f"✓ Correction NumPy: {fixed_files} fichiers corrigés")
    
    # Correction pour PyTorch 2.6+
    pytorch_fix_success = False
    try:
        from fix_pytorch_compatibility import fix_train_script
        pytorch_fix_success = fix_train_script(yolo_dir)
    except ImportError:
        print("⚠️ Module fix_pytorch_compatibility non trouvé.")
    
    if not pytorch_fix_success:
        print("Application manuelle de la correction PyTorch...")
        train_path = os.path.join(yolo_dir, 'train.py')
        
        try:
            with open(train_path, 'r') as f:
                content = f.read()
            
            if 'torch.load(weights, map_location=device)' in content and 'weights_only=False' not in content:
                modified_content = content.replace(
                    'torch.load(weights, map_location=device)',
                    'torch.load(weights, map_location=device, weights_only=False)'
                )
                
                with open(train_path, 'w') as f:
                    f.write(modified_content)
                print(f"✓ Correction PyTorch appliquée à {train_path}")
                pytorch_fix_success = True
            elif 'ckpt = torch.load(weights, map_location=device)' in content and 'weights_only=False' not in content:
                modified_content = content.replace(
                    'ckpt = torch.load(weights, map_location=device)',
                    'ckpt = torch.load(weights, map_location=device, weights_only=False)'
                )
                
                with open(train_path, 'w') as f:
                    f.write(modified_content)
                print(f"✓ Correction PyTorch appliquée à {train_path}")
                pytorch_fix_success = True
            else:
                print("⚠️ Le motif à corriger n'a pas été trouvé.")
                print("Recherche de toutes les occurrences de torch.load...")
                
                found = False
                for i, line in enumerate(content.split('\n')):
                    if 'torch.load' in line and 'weights' in line:
                        print(f"Ligne {i+1}: {line}")
                        found = True
                
                if not found:
                    print("⚠️ Aucune occurrence de torch.load avec weights trouvée.")
        except Exception as e:
            print(f"⚠️ Erreur lors de la correction PyTorch: {e}")
    
    if not pytorch_fix_success:
        print("\n⚠️ ATTENTION: La correction PyTorch n'a pas pu être appliquée!")
        print("Vous devrez modifier manuellement le fichier train.py avant de lancer l'entraînement.")
    
    # 7. Ajouter au PYTHONPATH
    if '/content' not in sys.path:
        sys.path.insert(0, '/content')
        print("✓ Répertoire /content ajouté au PYTHONPATH")
    
    # 8. Installer les dépendances YOLOv5-Face
    print("\n=== Installation des dépendances YOLOv5-Face ===")
    requirements_txt = os.path.join(yolo_dir, 'requirements.txt')
    if os.path.exists(requirements_txt):
        subprocess.run(['pip', 'install', '-r', requirements_txt, '--no-deps'], check=True)
        print("✓ Dépendances installées depuis requirements.txt")
    else:
        print("⚠️ Fichier requirements.txt non trouvé. Installation des dépendances essentielles...")
        subprocess.run(['pip', 'install', 'opencv-python>=4.5.0', 'PyYAML>=6.0', 'tqdm>=4.64.0'], check=True)
    
    print("\n" + "=" * 80)
    print("INSTALLATION TERMINÉE!")
    print("=" * 80)
    
    print("\nVous pouvez maintenant lancer l'entraînement avec:")
    print("python main.py --batch-size 16 --epochs 100 --img-size 640 --model-size s")
    
    return True

if __name__ == "__main__":
    setup_colab_environment()
