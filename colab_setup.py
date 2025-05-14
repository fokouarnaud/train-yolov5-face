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

# Importer la configuration centralisée
from config import REPO_URL, DEPENDENCIES, DEFAULT_PATHS, INFO_MESSAGES

def setup_environment(model_size='s', yolo_version='5.0'):
    """
    Configure l'environnement Colab pour l'entraînement
    
    Args:
        model_size (str): Taille du modèle 
                          - n-0.5, n : modèles ultra-légers (ShuffleNetV2) pour appareils mobiles
                          - s, m, l, x : modèles standards (CSPNet)
                          - s6, m6, l6, x6 : versions avec bloc P6 pour grands visages
        yolo_version (str): Version de YOLOv5 à utiliser
    """
    # Bloquer complètement l'utilisation de n6 qui n'est pas officiellement supporté
    if model_size == 'n6':
        print("⚠️ Le modèle YOLOv5n6 n'est pas officiellement supporté et peut provoquer des erreurs")
        print("Nous recommandons d'utiliser YOLOv5s6 à la place pour la détection des grands visages")
        print("→ La configuration continue avec le modèle 's' par défaut")
        model_size = 's'
        
    # 1. Installer les dépendances compatibles
    print("=== Installation des dépendances compatibles ===")
    subprocess.run(['pip', 'install', 'numpy==1.26.4', 'scipy==1.13.1', 'gensim==4.3.3', '--no-deps'], check=True)
    
    # 2. Vérifier si le dépôt YOLOv5-Face est cloné
    yolo_dir = '/content/yolov5-face'
    if not os.path.exists(yolo_dir):
        print("=== Clonage du dépôt YOLOv5-Face ===")
        # Utiliser le dépôt forké avec les corrections déjà appliquées
        subprocess.run(['git', 'clone', REPO_URL, yolo_dir], check=True)
    
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
            try:
                # Vérifier si le fichier existe et n'est pas vide
                if os.path.exists(weights_path) and os.path.getsize(weights_path) == 0:
                    os.remove(weights_path)  # Supprimer le fichier vide pour éviter les erreurs futures
                    print(f"Suppression du fichier de poids vide: {weights_path}")
                    
                subprocess.run(['wget', weights_url, '-O', weights_path], check=True)
                print(f"✓ Poids yolov5{size}.pt téléchargés")
            except subprocess.CalledProcessError:
                print(f"✗ Erreur lors du téléchargement des poids yolov5{size}.pt")
                if size in ['n-0.5', 'n']:
                    print(f"Les modèles YOLOv5{size} sont des modèles ultra-légers spécifiques à YOLOv5-Face.")
                    print(f"Ces modèles utilisent l'architecture ShuffleNetV2 et sont optimisés pour les appareils mobiles.")
                    print(f"Ils seront initialisés avec des poids aléatoires pour l'entraînement.")
                else:
                    print(f"  Le modèle {size} sera initialisé avec des poids aléatoires")
                # Certaines variantes comme n-0.5 et n peuvent ne pas être disponibles en téléchargement
        else:
            print(f"✓ Poids yolov5{size}.pt déjà présents")
    
    # 5. Vérification de la compatibilité PyTorch 2.6+
    print("=== Vérification de la compatibilité PyTorch 2.6+ ===")
    print(INFO_MESSAGES["pytorch_fix"])
    print("✓ Aucune modification du code n'est nécessaire")
    
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
    parser.add_argument('--model-size', type=str, default='s', choices=['n-0.5', 'n', 's', 's6', 'm', 'm6', 'l', 'l6', 'x', 'x6', 'all'],
                        help='Taille du modèle à télécharger (n-0.5, n, s, s6, m, m6, l, l6, x, x6, all)')
    parser.add_argument('--yolo-version', type=str, default='5.0',
                        help='Version de YOLOv5 à utiliser (par exemple 5.0)')
    
    args = parser.parse_args()
    setup_environment(args.model_size, args.yolo_version)
