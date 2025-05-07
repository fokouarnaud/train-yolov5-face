#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script d'installation rapide pour le projet YOLOv5-Face sur Google Colab
Ce script télécharge tous les fichiers nécessaires depuis GitHub
"""

import os
import subprocess
import sys

# Liste des fichiers Python à télécharger
python_files = [
    "main.py",
    "data_preparation.py",
    "model_training.py",
    "model_evaluation.py",
    "utils.py",
    "colab_setup.py",
    "patch_train_script.py"
]

# URL de base du répertoire GitHub (à remplacer par votre propre dépôt)
BASE_URL = "https://raw.githubusercontent.com/votre-utilisateur/yolov5-face-trainer/main/"

def download_project_files():
    """Télécharge tous les fichiers Python du projet depuis GitHub"""
    print("=== Téléchargement des fichiers Python du projet ===")
    
    for filename in python_files:
        url = f"{BASE_URL}{filename}"
        
        # Télécharger le fichier
        try:
            subprocess.run(['wget', url, '-O', filename], check=True)
            print(f"✓ Fichier {filename} téléchargé avec succès")
        except subprocess.CalledProcessError:
            print(f"✗ Erreur lors du téléchargement de {filename}")
            return False
    
    # Rendre les scripts exécutables
    subprocess.run(['chmod', '+x', '*.py'], check=True)
    
    return True

def main():
    """Fonction principale"""
    # Vérifier si nous sommes dans Colab
    try:
        import google.colab
        is_colab = True
    except ImportError:
        is_colab = False
    
    if not is_colab:
        print("⚠️ Ce script est conçu pour être exécuté dans Google Colab.")
        print("Si vous n'êtes pas dans Colab, certaines fonctionnalités pourraient ne pas fonctionner.")
    
    # Télécharger les fichiers Python du projet
    if download_project_files():
        print("\n✓ Installation réussie!")
        print("\nVous pouvez maintenant exécuter la configuration avec:")
        print("!python colab_setup.py")
        print("\nPuis lancer l'entraînement avec:")
        print("!python main.py")
    else:
        print("\n✗ L'installation a échoué.")
        print("Vérifiez votre connexion Internet et réessayez.")

if __name__ == "__main__":
    main()
