#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de configuration pour Google Colab
qui prépare l'environnement pour l'entraînement de YOLOv5-Face
"""

import os
import sys
import subprocess
import argparse

def setup_environment(model_size='s', yolo_version='5.0'):
    """
    Configure l'environnement Colab pour l'entraînement
    
    Args:
        model_size (str): Taille du modèle ('s', 'm', 'l', 'x')
        yolo_version (str): Version de YOLOv5 à utiliser
    """
    # 1. Installer les dépendances compatibles
    print("=== Installation des dépendances compatibles ===")
    subprocess.run(['pip', 'install', 'numpy==1.26.4', 'scipy==1.13.1', 'gensim==4.3.3', '--no-deps'], check=True)
    
    # 2. Vérifier si le dépôt YOLOv5-Face est cloné
    yolo_dir = '/content/yolov5-face'
    if not os.path.exists(yolo_dir):
        print("=== Clonage du dépôt YOLOv5-Face ===")
        subprocess.run(['git', 'clone', 'https://github.com/deepcam-cn/yolov5-face.git', yolo_dir], check=True)
    
    # 3. Créer le répertoire des poids
    weights_dir = os.path.join(yolo_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    
    # 4. Télécharger les poids pré-entraînés
    print(f"=== Téléchargement des poids YOLOv5 v{yolo_version} ===")
    weights_to_download = ['s', 'm', 'l', 'x'] if model_size == 'all' else [model_size]
    
    for size in weights_to_download:
        weights_url = f'https://github.com/ultralytics/yolov5/releases/download/v{yolo_version}/yolov5{size}.pt'
        weights_path = os.path.join(weights_dir, f'yolov5{size}.pt')
        
        if not os.path.exists(weights_path):
            print(f"Téléchargement de yolov5{size}.pt...")
            subprocess.run(['wget', weights_url, '-O', weights_path], check=True)
            print(f"✓ Poids yolov5{size}.pt téléchargés")
        else:
            print(f"✓ Poids yolov5{size}.pt déjà présents")
    
    # 5. Patcher le script train.py pour PyTorch 2.6+
    print("=== Modification du script train.py pour PyTorch 2.6+ ===")
    train_path = os.path.join(yolo_dir, 'train.py')
    
    if os.path.exists(train_path):
        # Lire le contenu du fichier
        with open(train_path, 'r') as f:
            content = f.read()
        
        # Remplacer la ligne torch.load
        modified_content = content.replace(
            'torch.load(weights, map_location=device)',
            'torch.load(weights, map_location=device, weights_only=False)'
        )
        
        # Vérifier si le remplacement a été effectué
        if modified_content == content:
            print(f"\nⓘ ATTENTION: Aucun changement n'a été effectué dans le fichier {train_path}!")
            print("Le motif 'torch.load(weights, map_location=device)' n'a pas été trouvé.")
            print("Vérifions s'il existe d'autres formats...")
            
            # Essayer d'autres formats possibles
            patterns_to_try = [
                ('ckpt = torch.load(weights, map_location=device)', 'ckpt = torch.load(weights, map_location=device, weights_only=False)'),
                ('torch.load(weights,map_location=device)', 'torch.load(weights,map_location=device, weights_only=False)'),
                ('torch.load( weights, map_location=device )', 'torch.load( weights, map_location=device, weights_only=False )')
            ]
            
            for old_pattern, new_pattern in patterns_to_try:
                if old_pattern in content:
                    modified_content = content.replace(old_pattern, new_pattern)
                    print(f"✓ Motif alternatif trouvé et remplacé: '{old_pattern}'")
                    break
        
        # Sauvegarder le fichier modifié
        with open(train_path, 'w') as f:
            f.write(modified_content)
        print(f"✓ Fichier {train_path} modifié avec succès")
    else:
        print(f"✗ Fichier {train_path} non trouvé")
    
    # 6. Ajouter le répertoire courant au PYTHONPATH
    if '/content' not in sys.path:
        print("=== Configuration du PYTHONPATH ===")
        sys.path.insert(0, '/content')
        print("✓ Répertoire /content ajouté au PYTHONPATH")
    
    # 7. Vérifier la présence des scripts Python
    scripts = ['main.py', 'data_preparation.py', 'model_training.py', 'model_evaluation.py', 'utils.py']
    missing_scripts = [script for script in scripts if not os.path.exists(f'/content/{script}')]
    
    if missing_scripts:
        print(f"⚠️ Attention: Les scripts suivants sont manquants: {', '.join(missing_scripts)}")
        print("Assurez-vous de les copier depuis Google Drive ou de les télécharger.")
    else:
        print("✓ Tous les scripts Python nécessaires sont présents")
    
    print("\n=== Configuration terminée ===")
    print("Vous pouvez maintenant exécuter le script principal avec la commande:")
    print("!python main.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configuration de l'environnement Colab pour YOLOv5-Face")
    parser.add_argument('--model-size', type=str, default='s', choices=['s', 'm', 'l', 'x', 'all'],
                        help='Taille du modèle à télécharger (s, m, l, x, all)')
    parser.add_argument('--yolo-version', type=str, default='5.0',
                        help='Version de YOLOv5 à utiliser (par exemple 5.0)')
    
    args = parser.parse_args()
    setup_environment(args.model_size, args.yolo_version)
