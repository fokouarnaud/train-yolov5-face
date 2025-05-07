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

# Copier les scripts depuis Google Drive
subprocess.run(['cp', '/content/drive/MyDrive/yolov5_face_scripts/*.py', '/content/'])

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

subprocess.run(['git', 'clone', 'https://github.com/deepcam-cn/yolov5-face.git', '/content/yolov5-face'])

# Créer le répertoire des poids et télécharger les poids pré-entraînés
subprocess.run(['mkdir', '-p', '/content/yolov5-face/weights'])
subprocess.run(['wget', '-O', '/content/yolov5-face/weights/yolov5s.pt', 
                'https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt'])

# Étape cruciale: Correction de compatibilité PyTorch 2.6+
print("\n" + "=" * 70)
print(" CORRECTION POUR PYTORCH 2.6+ ".center(70, "="))
print("=" * 70 + "\n")

train_path = '/content/yolov5-face/train.py'

# Vérifier si le fichier existe
if not os.path.exists(train_path):
    print(f"❌ ERREUR: Fichier {train_path} non trouvé!")
    sys.exit(1)

# Lire le contenu du fichier
with open(train_path, 'r') as f:
    content = f.read()

# Vérifier si le fichier a déjà été corrigé
if 'weights_only=False' in content:
    print("✅ Le fichier train.py est déjà compatible avec PyTorch 2.6+")
else:
    # Chercher spécifiquement la ligne qui cause l'erreur (celle mentionnée dans votre message d'erreur)
    if 'ckpt = torch.load(weights, map_location=device)' in content:
        modified_content = content.replace(
            'ckpt = torch.load(weights, map_location=device)',
            'ckpt = torch.load(weights, map_location=device, weights_only=False)'
        )
        
        # Sauvegarder le fichier modifié
        with open(train_path, 'w') as f:
            f.write(modified_content)
        print("✅ Ligne 'ckpt = torch.load(weights, map_location=device)' corrigée avec succès!")
        
        # Vérifier que la modification a bien été appliquée
        with open(train_path, 'r') as f:
            check_content = f.read()
        
        if 'weights_only=False' in check_content:
            print("✅ Vérification réussie: Le paramètre weights_only=False a bien été ajouté.")
        else:
            print("❌ ERREUR: La modification n'a pas été correctement enregistrée!")
            print("Tentative de modification manuelle requise.")
    else:
        print("🔍 La ligne exacte 'ckpt = torch.load(weights, map_location=device)' n'a pas été trouvée.")
        print("Recherche d'autres formats possibles...")
        
        # Rechercher d'autres formats possibles
        found_match = False
        
        # Approche 1: Expressions régulières pour trouver les variantes
        pattern = r'(ckpt\s*=\s*torch\.load\s*\(\s*weights\s*,\s*map_location\s*=\s*device\s*\))'
        match = re.search(pattern, content)
        
        if match:
            found_text = match.group(1)
            replacement = found_text.replace(')', ', weights_only=False)')
            modified_content = content.replace(found_text, replacement)
            
            # Sauvegarder le fichier modifié
            with open(train_path, 'w') as f:
                f.write(modified_content)
            
            print(f"✅ Correction appliquée via regex: '{found_text}' → '{replacement}'")
            found_match = True
        
        # Approche 2: Analyse ligne par ligne si l'approche 1 échoue
        if not found_match:
            print("🔍 Analyse ligne par ligne du fichier train.py...")
            lines = content.split('\n')
            
            for i, line in enumerate(lines):
                if 'torch.load' in line and 'weights' in line and 'map_location' in line:
                    print(f"Ligne trouvée ({i+1}): {line}")
                    
                    # Rechercher la parenthèse fermante
                    close_paren_pos = line.rfind(')')
                    if close_paren_pos > 0:
                        modified_line = line[:close_paren_pos] + ', weights_only=False' + line[close_paren_pos:]
                        lines[i] = modified_line
                        print(f"✅ Ligne modifiée: {modified_line}")
                        
                        # Reconstruire le contenu du fichier
                        modified_content = '\n'.join(lines)
                        
                        # Sauvegarder le fichier modifié
                        with open(train_path, 'w') as f:
                            f.write(modified_content)
                        
                        found_match = True
                        break
            
        # Si aucune méthode n'a fonctionné, afficher toutes les lignes contenant torch.load
        if not found_match:
            print("\n❌ ERREUR: Aucune méthode automatique n'a pu corriger le fichier!")
            print("Voici toutes les lignes contenant 'torch.load' dans le fichier:")
            
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'torch.load' in line:
                    print(f"Ligne {i+1}: {line}")
            
            print("\n⚠️ ATTENTION: Vous devez corriger manuellement le fichier train.py!")
            print("Commande pour éditer le fichier: !nano /content/yolov5-face/train.py")
            
            # Demander à l'utilisateur s'il souhaite continuer
            user_input = input("\nVoulez-vous continuer malgré tout? (oui/non): ")
            if user_input.lower() not in ['oui', 'o', 'yes', 'y']:
                print("Programme interrompu par l'utilisateur.")
                sys.exit(1)

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
