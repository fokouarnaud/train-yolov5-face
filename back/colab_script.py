#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script tout-en-un optimisé pour Google Colab pour YOLOv5-Face
Ce script intègre toutes les étapes nécessaires:
1. Configuration de l'environnement
2. Installation des dépendances
3. Clonage du dépôt
4. Correction des problèmes de compatibilité PyTorch
5. Correction de la fonction de perte
6. Entraînement du modèle
7. Sauvegarde des résultats
"""

# Étape 1: Monter Google Drive et copier les scripts
from google.colab import drive
drive.mount('/content/drive')

# Créer un répertoire pour les scripts
!mkdir -p /content
!cp /content/drive/MyDrive/yolov5_face_scripts/{main.py,data_preparation.py,model_training.py,model_evaluation.py,utils.py,colab_setup.py} /content/

# Étape 2: Installer les dépendances compatibles
!pip install numpy==1.26.4 scipy==1.13.1 gensim==4.3.3 --no-deps
!pip install torch>=2.0.0 torchvision>=0.15.0
!pip install opencv-python  # S'assurer que OpenCV est installé

# Étape 3: Exécuter le script de configuration
%cd /content
!python colab_setup.py --model-size s

# Corriger la fonction de perte dans le fichier loss.py
def fix_loss_file():
    """
    Vérifie et corrige la fonction de perte dans le fichier loss.py
    """
    import os
    
    loss_file = '/content/yolov5-face/utils/loss.py'
    
    # Vérifier si le fichier existe
    if not os.path.exists(loss_file):
        print(f"❌ ERREUR: Le fichier {loss_file} n'existe pas!")
        return False
    
    with open(loss_file, 'r') as f:
        content = f.read()
    
    # Vérifier si .long() est déjà présent
    if 'gj.clamp_(0, gain[3] - 1).long()' in content:
        print("✅ Le fichier loss.py a déjà la correction .long()")
        return True
    
    # Si non, appliquer la correction
    old_text = 'indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))'
    new_text = 'indices.append((b, a, gj.clamp_(0, gain[3] - 1).long(), gi.clamp_(0, gain[2] - 1).long()))'
    
    if old_text in content:
        modified_content = content.replace(old_text, new_text)
        
        with open(loss_file, 'w') as f:
            f.write(modified_content)
        
        print("✅ Correction appliquée avec succès au fichier loss.py")
        return True
    else:
        print("⚠️ Format attendu non trouvé dans loss.py, vérification manuelle recommandée")
        return False

# Vérifier la compatibilité PyTorch dans train.py
def fix_pytorch_compatibility():
    """
    Ajoute le paramètre weights_only=False à torch.load dans train.py
    """
    import os
    import re
    
    train_file = '/content/yolov5-face/train.py'
    
    if not os.path.exists(train_file):
        print(f"❌ ERREUR: Le fichier {train_file} n'existe pas!")
        return False
    
    with open(train_file, 'r') as f:
        content = f.read()
    
    # Vérifier si la correction est déjà appliquée
    if 'weights_only=False' in content:
        print("✅ Le fichier train.py a déjà le paramètre weights_only=False")
        return True
    
    # Appliquer la correction
    old_text = 'ckpt = torch.load(weights, map_location=device)'
    new_text = 'ckpt = torch.load(weights, map_location=device, weights_only=False)'
    
    if old_text in content:
        modified_content = content.replace(old_text, new_text)
        
        with open(train_file, 'w') as f:
            f.write(modified_content)
        
        print("✅ Correction PyTorch appliquée avec succès à train.py")
        return True
    else:
        # Recherche avec expression régulière si le format exact n'est pas trouvé
        pattern = r'(ckpt\s*=\s*torch\.load\s*\(\s*weights\s*,\s*map_location\s*=\s*device\s*\))'
        match = re.search(pattern, content)
        
        if match:
            found_text = match.group(1)
            replacement = found_text.replace(')', ', weights_only=False)')
            modified_content = content.replace(found_text, replacement)
            
            with open(train_file, 'w') as f:
                f.write(modified_content)
            
            print(f"✅ Correction PyTorch appliquée via regex")
            return True
        else:
            print("⚠️ Format attendu non trouvé dans train.py, vérification manuelle recommandée")
            return False

# Installer werkzeug pour résoudre le problème de TensorBoard
!pip install werkzeug

# Appliquer les corrections
print("\n=== Application des corrections avant l'entraînement ===")
fix_loss_file()
fix_pytorch_compatibility()

# Étape 4: Lancer l'entraînement
!python main.py

# Étape 5: Visualiser les résultats
%load_ext tensorboard
%tensorboard --logdir /content/yolov5-face/runs/train/face_detection_transfer

# Étape 6: Créer un dossier pour les résultats
!mkdir -p /content/drive/MyDrive/YOLOv5_Face_Results
# Copier les résultats de l'entraînement vers Google Drive
!cp -r /content/yolov5-face/runs/train/face_detection_transfer /content/drive/MyDrive/YOLOv5_Face_Results/
