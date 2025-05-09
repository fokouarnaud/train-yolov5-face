#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script tout-en-un optimisé pour Google Colab pour YOLOv5-Face
Ce script intègre toutes les étapes nécessaires:
1. Configuration de l'environnement
2. Installation des dépendances
3. Correction des problèmes de compatibilité PyTorch
4. Correction de la fonction de perte
5. Entraînement du modèle
6. Visualisation et sauvegarde des résultats
"""

# Étape 1: Monter Google Drive et copier les scripts
from google.colab import drive
drive.mount('/content/drive')

# Créer un répertoire pour les scripts
!mkdir -p /content
!cp /content/drive/MyDrive/yolov5_face_scripts/{main.py,data_preparation.py,model_training.py,model_evaluation.py,utils.py,colab_setup.py,pytorch_fix.py,fix_loss.py} /content/

# Étape 2: Installer les dépendances compatibles
!pip install numpy==1.26.4 scipy==1.13.1 gensim==4.3.3 --no-deps
!pip install torch>=2.0.0 torchvision>=0.15.0
!pip install opencv-python  # S'assurer que OpenCV est installé

# Étape 3: Exécuter le script de configuration
%cd /content
!python colab_setup.py --model-size s

# Étape 3.5: Corriger la compatibilité PyTorch pour train.py
print("\n=== Correction de la compatibilité PyTorch pour train.py ===")
!python pytorch_fix.py

# Étape 3.6: Corriger le problème de conversion de type dans loss.py
print("\n=== Correction du problème de conversion de type dans loss.py ===")
!python fix_loss.py

# Installer werkzeug pour résoudre le problème de TensorBoard
print("\n=== Installation de werkzeug pour TensorBoard ===")
!pip install werkzeug

# Vérification des corrections avant de continuer
import os

def verify_pytorch_fix():
    """Vérifie que le correctif PyTorch a bien été appliqué"""
    train_path = '/content/yolov5-face/train.py'
    if not os.path.exists(train_path):
        print("❌ ERREUR: Le fichier train.py n'a pas été trouvé!")
        return False
        
    with open(train_path, 'r') as f:
        content = f.read()
        
    if 'weights_only=False' in content:
        print("✅ Vérification PyTorch: Le paramètre weights_only=False est correctement présent.")
        return True
    else:
        print("❌ AVERTISSEMENT: Le paramètre weights_only=False n'a pas été ajouté à train.py!")
        return False

def verify_loss_fix():
    """Vérifie que le correctif de la fonction de perte a bien été appliqué"""
    loss_path = '/content/yolov5-face/utils/loss.py'
    if not os.path.exists(loss_path):
        print("❌ ERREUR: Le fichier loss.py n'a pas été trouvé!")
        return False
        
    with open(loss_path, 'r') as f:
        content = f.read()
        
    if '.long()' in content and 'gj.clamp_(0, gain[3] - 1).long()' in content:
        print("✅ Vérification loss.py: La conversion .long() est correctement présente.")
        return True
    else:
        print("❌ AVERTISSEMENT: La conversion .long() n'a pas été ajoutée à loss.py!")
        return False

print("\n=== Vérification des corrections ===")
pytorch_check = verify_pytorch_fix()
loss_check = verify_loss_fix()

if not pytorch_check or not loss_check:
    print("\n⚠️ ATTENTION: Des problèmes ont été détectés avec les corrections.")
    response = input("Voulez-vous continuer malgré tout? (oui/non): ")
    if response.lower() not in ['oui', 'o', 'y', 'yes']:
        print("Exécution arrêtée par l'utilisateur.")
        import sys
        sys.exit(1)

# Étape 4: Lancer l'entraînement
print("\n=== Lancement de l'entraînement ===")
!python main.py

# Étape 5: Visualiser les résultats
print("\n=== Configuration de TensorBoard pour visualiser les résultats ===")
%load_ext tensorboard
%tensorboard --logdir /content/yolov5-face/runs/train/face_detection_transfer

# Étape 6: Créer un dossier pour les résultats et les sauvegarder
print("\n=== Sauvegarde des résultats vers Google Drive ===")
!mkdir -p /content/drive/MyDrive/YOLOv5_Face_Results
# Copier les résultats de l'entraînement vers Google Drive
!cp -r /content/yolov5-face/runs/train/face_detection_transfer /content/drive/MyDrive/YOLOv5_Face_Results/

print("\n✅ TERMINÉ: L'entraînement est terminé et les résultats ont été sauvegardés.")
print("Les résultats sont disponibles dans: /content/drive/MyDrive/YOLOv5_Face_Results/")
