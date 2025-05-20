#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module pour l'entraînement du modèle YOLOv5-Face
"""

import os
import subprocess
import yaml

# Importer la configuration centralisée
from config import DEFAULT_TRAINING, DEFAULT_PATHS, MODEL_CONFIGS

class ModelTrainer:
    """Classe pour gérer l'entraînement du modèle YOLOv5-Face"""
    
    def __init__(self, yolo_dir, data_dir, batch_size=DEFAULT_TRAINING["batch_size"], epochs=DEFAULT_TRAINING["epochs"], img_size=DEFAULT_TRAINING["img_size"], model_size=DEFAULT_TRAINING["model_size"], yolo_version=DEFAULT_TRAINING["yolo_version"]):
        """Initialise la classe d'entraînement du modèle
        
        Args:
            yolo_dir (str): Répertoire de YOLOv5-Face
            data_dir (str): Répertoire des données
            batch_size (int): Taille du batch pour l'entraînement
            epochs (int): Nombre d'epochs d'entraînement
            img_size (int): Taille d'image pour l'entraînement
            model_size (str): Taille du modèle YOLOv5 (n-0.5, n, s, s6, m, m6, l, l6, x, x6)
                             Les modèles n-0.5 et n sont des modèles ultra-légers basés sur ShuffleNetV2
                             Les modèles avec suffixe '6' incluent un bloc de sortie P6 pour améliorer
                             la détection des grands visages
        """
        self.yolo_dir = yolo_dir
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.epochs = epochs
        self.img_size = img_size
        self.model_size = model_size
        self.yolo_version = yolo_version
    
    def download_weights(self):
        """Télécharge les poids pré-entraînés pour le modèle YOLOv5
        
        Notes:
            - Les modèles ultra-légers n-0.5 et n sont spécifiques à YOLOv5-Face et utilisent ShuffleNetV2
            - Les modèles standards (s, m, l, x) utilisent CSPNet comme backbone
            - Les variantes avec suffixe '6' incluent un bloc de sortie P6 pour les grands visages
            - Le modèle 'ad' est ADYOLOv5-Face, une version améliorée de YOLOv5s-Face pour les petits visages
        """
        print("\n=== Téléchargement des poids pré-entraînés ===")
        
        # Utiliser la configuration du modèle appropriée
        config = MODEL_CONFIGS.get(self.model_size, {})
        base_weights = config.get('weights', f'yolov5{self.model_size}.pt')
        
        # Chemin du répertoire des poids
        weights_dir = os.path.join(self.yolo_dir, 'weights')
        weights_path = os.path.join(weights_dir, base_weights)
        
        # Créer le répertoire des poids s'il n'existe pas
        os.makedirs(weights_dir, exist_ok=True)
        
        # Pour ADYOLOv5-Face, nous utilisons les poids de YOLOv5s comme base
        download_weights = base_weights
        
        # Téléchargement direct des poids
        weights_url = f'https://github.com/ultralytics/yolov5/releases/download/v{self.yolo_version}/{download_weights}'
        
        # Télécharger les poids
        try:
            subprocess.run(['wget', weights_url, '-O', weights_path], check=True)
            print(f"✓ Poids téléchargés: {weights_path}")
            
            # Message spécifique pour ADYOLOv5-Face
            if self.model_size == 'ad':
                print(f"  Note: Pour ADYOLOv5-Face, nous utilisons les poids de YOLOv5s comme base")
                print(f"  Ces poids seront adaptés à la nouvelle architecture pendant l'entraînement")
                
        except subprocess.CalledProcessError:
            print(f"✗ Erreur lors du téléchargement des poids depuis {weights_url}")
            if self.model_size in ['n-0.5', 'n']:
                print(f"Les modèles YOLOv5{self.model_size} sont des modèles ultra-légers spécifiques à YOLOv5-Face.")
                print(f"Ces modèles utilisent l'architecture ShuffleNetV2 et sont optimisés pour les appareils mobiles.")
            elif self.model_size.endswith('6'):
                print(f"Le modèle YOLOv5{self.model_size} inclut un bloc de sortie P6 pour améliorer la détection des grands visages.")
                if self.model_size == 'n6':
                    print(f"ATTENTION: Le modèle YOLOv5n6 n'est pas officiellement supporté. Utilisez YOLOv5s6 à la place.")
            elif self.model_size == 'ad':
                print(f"ADYOLOv5-Face est une version améliorée de YOLOv5s-Face avec un mécanisme GD et une tête supplémentaire pour les petits visages.")
                print(f"Nous utilisons les poids de YOLOv5s comme base pour l'entraînement.")
            else:
                print("Certaines variantes peuvent ne pas être disponibles dans les versions officielles.")
            print("L'entraînement continuera sans poids pré-entraînés ou avec des poids aléatoires.")
    
    def train(self):
        """Lance l'entraînement du modèle YOLOv5-Face
        
        Returns:
            bool: True si l'entraînement a réussi, False sinon
        """
        # Bloquer complètement l'utilisation de n6 qui n'est pas officiellement supporté
        if self.model_size == 'n6':
            print("⚠️ Le modèle YOLOv5n6 n'est pas officiellement supporté et provoque des erreurs")
            print("Utilisez plutôt YOLOv5s6 pour la détection des grands visages")
            print("→ Vous pouvez relancer avec --model-size s6")
            return False
            
        # Télécharger les poids pré-entraînés
        self.download_weights()
        
        print("\n=== Démarrage de l'entraînement ===")
        
        # Utiliser la configuration du modèle appropriée
        config = MODEL_CONFIGS.get(self.model_size, {})
        model_yaml = config.get('yaml', f'yolov5{self.model_size}.yaml')
        base_weights = config.get('weights', f'yolov5{self.model_size}.pt')
        
        # Vérifier que tous les fichiers nécessaires existent
        yaml_path = f'{self.yolo_dir}/data/widerface.yaml'
        weights_path = f'{self.yolo_dir}/weights/{base_weights}'
        cfg_path = f'{self.yolo_dir}/models/{model_yaml}'
        
        # Afficher les informations sur le modèle sélectionné
        print(f"Modèle sélectionné: {self.model_size}")
        print(f"  - Configuration YAML: {model_yaml}")
        print(f"  - Poids de base: {base_weights}")
        
        # Message spécial pour ADYOLOv5-Face
        if self.model_size == 'ad':
            print("\nℹ️ INFORMATIONS SUR ADYOLOV5-FACE")
            print("  - Architecture améliorée basée sur YOLOv5s-Face")
            print("  - Mécanisme Gather-and-Distribute (GD) dans le neck")
            print("  - Tête de détection supplémentaire pour les petits visages")
            print("  - Performances améliorées sur les visages de petite taille")
        
        if not os.path.exists(yaml_path):
            print(f"✗ Fichier de configuration non trouvé: {yaml_path}")
            return False
        
        if not os.path.exists(cfg_path):
            print(f"✗ Fichier de configuration du modèle non trouvé: {cfg_path}")
            print(f"  Assurez-vous que le fichier {model_yaml} existe dans {self.yolo_dir}/models/")
            return False
        
        # Pour les poids, nous pouvons continuer même s'ils ne sont pas trouvés (entraînement à partir de zéro)
        weights_arg = ''
        if os.path.exists(weights_path):
            # Vérifier que le fichier n'est pas vide
            if os.path.getsize(weights_path) > 0:
                weights_arg = weights_path
            else:
                print(f"⚠️ Fichier de poids vide détecté: {weights_path}")
                print("  Ce fichier sera supprimé et l'entraînement commencera à partir de poids aléatoires")
                os.remove(weights_path)  # Supprimer le fichier vide pour éviter les erreurs futures
        else:
            print(f"⚠️ Fichier de poids non trouvé: {weights_path}")
            print("  L'entraînement commencera à partir de poids aléatoires")
        
        # Construire la commande d'entraînement
        train_cmd = [
            'python', f'{self.yolo_dir}/train.py',
            '--data', yaml_path,
            '--cfg', cfg_path,
            '--weights', weights_arg,
            '--batch-size', str(self.batch_size),
            '--epochs', str(self.epochs),
            '--img', str(self.img_size),
            '--project', f'{self.yolo_dir}/runs/train',
            '--name', 'face_detection_transfer',
            '--exist-ok',
            '--cache',
            '--rect'
        ]
        
        # Utiliser le fichier d'hyperparamètres spécial pour ADYOLOv5-Face
        if self.model_size == 'ad':
            hyp_adyolo = f'{self.yolo_dir}/data/hyp.adyolo.yaml'
            if os.path.exists(hyp_adyolo):
                train_cmd.extend(['--hyp', hyp_adyolo])
                print(f"  - Utilisation des hyperparamètres optimisés pour petits visages: {hyp_adyolo}")
            else:
                print(f"  - Fichier d'hyperparamètres {hyp_adyolo} non trouvé, utilisation des valeurs par défaut")
        
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
