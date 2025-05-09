#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module pour l'entraînement du modèle YOLOv5-Face
"""

import os
import subprocess
import yaml

# Importer la configuration centralisée
from config import DEFAULT_TRAINING, DEFAULT_PATHS

class ModelTrainer:
    """Classe pour gérer l'entraînement du modèle YOLOv5-Face"""
    
    def __init__(self, yolo_dir, data_dir, batch_size=32, epochs=300, img_size=640, model_size='s', yolo_version='5.0'):
        """Initialise la classe d'entraînement du modèle
        
        Args:
            yolo_dir (str): Répertoire de YOLOv5-Face
            data_dir (str): Répertoire des données
            batch_size (int): Taille du batch pour l'entraînement
            epochs (int): Nombre d'epochs d'entraînement
            img_size (int): Taille d'image pour l'entraînement
            model_size (str): Taille du modèle YOLOv5 (s, m, l, x)
        """
        self.yolo_dir = yolo_dir
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.img_size = img_size
        self.model_size = model_size
        self.yolo_version = yolo_version
    
    def download_weights(self):
        """Télécharge les poids pré-entraînés pour le modèle YOLOv5"""
        print("\n=== Téléchargement des poids pré-entraînés ===")
        
        # Chemin du répertoire des poids
        weights_dir = os.path.join(self.yolo_dir, 'weights')
        weights_path = os.path.join(weights_dir, f'yolov5{self.model_size}.pt')
        
        # Créer le répertoire des poids s'il n'existe pas
        os.makedirs(weights_dir, exist_ok=True)
        
        # Téléchargement direct des poids
        weights_url = f'https://github.com/ultralytics/yolov5/releases/download/v{self.yolo_version}/yolov5{self.model_size}.pt'
        
        # Télécharger les poids
        subprocess.run(['wget', weights_url, '-O', weights_path], check=True)
        print(f"✓ Poids téléchargés: {weights_path}")
    
    def train(self):
        """Lance l'entraînement du modèle YOLOv5-Face
        
        Returns:
            bool: True si l'entraînement a réussi, False sinon
        """
        # Télécharger les poids pré-entraînés
        self.download_weights()
        
        print("\n=== Démarrage de l'entraînement ===")
        
        # Vérifier que tous les fichiers nécessaires existent
        yaml_path = f'{self.yolo_dir}/data/widerface.yaml'
        weights_path = f'{self.yolo_dir}/weights/yolov5{self.model_size}.pt'
        
        if not os.path.exists(yaml_path):
            print(f"✗ Fichier de configuration non trouvé: {yaml_path}")
            return False
        
        if not os.path.exists(weights_path):
            print(f"✗ Fichier de poids non trouvé: {weights_path}")
            return False
        
        # Construire la commande d'entraînement
        train_cmd = [
            'python', f'{self.yolo_dir}/train.py',
            '--data', yaml_path,
            '--cfg', f'{self.yolo_dir}/models/yolov5{self.model_size}.yaml',
            '--weights', weights_path,
            '--batch-size', str(self.batch_size),
            '--epochs', str(self.epochs),
            '--img', str(self.img_size),
            '--project', f'{self.yolo_dir}/runs/train',
            '--name', 'face_detection_transfer',
            '--exist-ok',
            '--cache',
            '--rect'
        ]
        
        # Lancer l'entraînement
        print("Commande d'entraînement:")
        print(' '.join(train_cmd))
        
        try:
            subprocess.run(train_cmd, check=True)
            print("\n✓ Entraînement terminé avec succès!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Erreur lors de l'entraînement: {e}")
            return False
