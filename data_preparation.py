#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module pour la préparation des données du dataset WIDER Face
pour l'entraînement de YOLOv5-Face sur Google Colab
"""

import os
import zipfile
import shutil
import traceback
import cv2
import numpy as np
import yaml
from google.colab import drive
from pathlib import Path

# Importer la configuration centralisée
from config import DEFAULT_PATHS

class DataPreparation:
    """Classe pour gérer la préparation des données pour YOLOv5-Face"""
    
    def __init__(self, root_dir, yolo_dir, data_dir, drive_dataset_path):
        """Initialise la classe de préparation des données
        
        Args:
            root_dir (str): Répertoire racine de Colab
            yolo_dir (str): Répertoire de YOLOv5-Face
            data_dir (str): Répertoire des données
            drive_dataset_path (str): Chemin vers les données sur Google Drive
        """
        self.root_dir = root_dir
        self.yolo_dir = yolo_dir
        self.data_dir = data_dir
        self.drive_dataset_path = drive_dataset_path
        
        # Vérifier si le dossier drive est monté
        self.drive_mounted = os.path.exists('/content/drive')
        
        # Chemins des fichiers ZIP
        self.zip_files = {
            f'{self.drive_dataset_path}/WIDER_train.zip': f'{self.data_dir}/tmp/',
            f'{self.drive_dataset_path}/WIDER_val.zip': f'{self.data_dir}/tmp/',
            f'{self.drive_dataset_path}/WIDER_test.zip': f'{self.data_dir}/tmp/',
            f'{self.drive_dataset_path}/retinaface_gt.zip': f'{self.data_dir}/tmp/'
        }
        
        # Structure des répertoires à créer
        self.directories = [
            f'{self.data_dir}/tmp/train/images',
            f'{self.data_dir}/tmp/val/images',
            f'{self.data_dir}/tmp/test/images',
            f'{self.data_dir}/train/images',
            f'{self.data_dir}/train/labels',
            f'{self.data_dir}/val/images',
            f'{self.data_dir}/val/labels',
            f'{self.data_dir}/test/images',
            f'{self.data_dir}/test/labels'
        ]
    
    def prepare_dataset(self):
        """Exécute toutes les étapes de préparation des données"""
        try:
            # Monter Google Drive si ce n'est pas déjà fait
            if not self.drive_mounted:
                print("Montage de Google Drive...")
                drive.mount('/content/drive')
                self.drive_mounted = True
            
            # Créer les répertoires
            self._create_directories()
            
            # Extraire les datasets
            self._extract_datasets()
            
            # Copier les images
            self._copy_images()
            
            # Convertir les annotations
            self._convert_annotations()
            
            # Filtrer les images corrompues
            self._filter_corrupted_images()
            
            # Créer le fichier YAML
            self._create_yaml_config()
            
            print("\n✓ Préparation du dataset terminée avec succès")
            return True
            
        except Exception as e:
            print(f"✗ Erreur lors de la préparation du dataset: {e}")
            print(traceback.format_exc())
            return False
    
    def _create_directories(self):
        """Crée les répertoires nécessaires pour le dataset"""
        print("\n=== Création des répertoires ===")
        
        for directory in self.directories:
            os.makedirs(directory, exist_ok=True)
            print(f"✓ Création du répertoire: {directory}")
    
    def _extract_datasets(self):
        """Extrait les fichiers zip du dataset"""
        print("\n=== Extraction des datasets ===")
        
        for zip_path, extract_path in self.zip_files.items():
            if os.path.exists(zip_path):
                print(f"Extraction de {os.path.basename(zip_path)}...")
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_path)
                    print(f"✓ Extraction réussie: {zip_path}")
                except Exception as e:
                    print(f"✗ Erreur lors de l'extraction de {zip_path}: {e}")
            else:
                print(f"✗ Fichier non trouvé: {zip_path}")
    
    def _copy_images(self):
        """Copie les images vers la structure finale"""
        print("\n=== Copie des images vers la structure finale ===")
        
        sources = {
            'train_images': f'{self.data_dir}/tmp/WIDER_train/images',
            'val_images': f'{self.data_dir}/tmp/WIDER_val/images',
            'test_images': f'{self.data_dir}/tmp/WIDER_test/images',
        }
        
        destinations = {
            'train_images': f'{self.data_dir}/train/images/',
            'val_images': f'{self.data_dir}/val/images/',
            'test_images': f'{self.data_dir}/test/images/',
        }
        
        for src_name, src_path in sources.items():
            dst_path = destinations[src_name]
            
            if os.path.exists(src_path):
                print(f"Copie de {src_path} vers {dst_path}")
                
                # Parcourir tous les fichiers et dossiers
                for root, dirs, files in os.walk(src_path):
                    rel_path = os.path.relpath(root, src_path)
                    if rel_path == '.':
                        rel_path = ''
                    
                    # Créer les répertoires
                    for d in dirs:
                        os.makedirs(os.path.join(dst_path, rel_path, d), exist_ok=True)
                    
                    # Copier les fichiers
                    for f in files:
                        src_file = os.path.join(root, f)
                        dst_file = os.path.join(dst_path, rel_path, f)
                        os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                        shutil.copy2(src_file, dst_file)
                
                print(f"✓ Images copiées de {src_path} vers {dst_path}")
            else:
                print(f"✗ Source {src_path} non trouvée")
    
    def _convert_annotations(self):
        """Convertit les annotations du format WIDER Face au format YOLO"""
        print("\n=== Conversion des annotations ===")
        
        label_paths = {
            'train': f'{self.data_dir}/tmp/train/label.txt',
            'val': f'{self.data_dir}/tmp/val/label.txt',
        }
        
        for set_name, label_path in label_paths.items():
            images_dir = f'{self.data_dir}/{set_name}/images'
            labels_dir = f'{self.data_dir}/{set_name}/labels'
            
            if os.path.exists(label_path):
                print(f"Traitement des annotations {set_name}...")
                try:
                    self._process_label_file(label_path, images_dir, labels_dir)
                except Exception as e:
                    print(f"✗ Erreur lors de la conversion des annotations {set_name}: {e}")
                    print(traceback.format_exc())
            else:
                print(f"✗ Fichier d'annotations non trouvé: {label_path}")
    
    def _process_label_file(self, label_path, images_dir, labels_dir):
        """Traite le fichier d'annotations et convertit au format YOLO"""
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        current_image = None
        current_labels = []
        processed_files = 0
        skipped_files = 0
        
        for line in lines:
            line = line.strip()
            
            if line.startswith('#'):
                # Si nous avons déjà une image en cours de traitement, sauvegardons ses annotations
                if current_image is not None:
                    success = self._save_yolo_annotations(current_image, current_labels, images_dir, labels_dir)
                    if success:
                        processed_files += 1
                    else:
                        skipped_files += 1
                
                # Nouvelle image
                image_path = line[2:]
                current_image = image_path
                current_labels = []
            else:
                # Ajout d'une annotation
                values = line.split()
                if len(values) >= 4:  # Au moins x, y, w, h
                    current_labels.append(values)
        
        # Traiter la dernière image
        if current_image is not None:
            success = self._save_yolo_annotations(current_image, current_labels, images_dir, labels_dir)
            if success:
                processed_files += 1
            else:
                skipped_files += 1
        
        print(f"✓ Conversion terminée: {processed_files} fichiers traités, {skipped_files} fichiers ignorés")
    
    def _save_yolo_annotations(self, image_path, labels, images_dir, labels_dir):
        """Sauvegarde les annotations au format YOLO pour YOLOv5-face"""
        try:
            # Déterminer le chemin complet de l'image
            img_path = os.path.join(images_dir, image_path)
            
            # Vérifier si l'image existe
            if not os.path.exists(img_path):
                return False
            
            # Lire l'image pour obtenir ses dimensions
            img = cv2.imread(img_path)
            if img is None:
                return False
            
            height, width, _ = img.shape
            
            # Créer le fichier d'annotations YOLO
            base_name = os.path.splitext(image_path)[0]
            label_path = os.path.join(labels_dir, base_name + '.txt')
            
            # Créer le répertoire parent si nécessaire
            os.makedirs(os.path.dirname(label_path), exist_ok=True)
            
            valid_annotations = []
            
            for label in labels:
                # Format YOLO: class x_center y_center width height
                try:
                    x = float(label[0])
                    y = float(label[1])
                    w = float(label[2])
                    h = float(label[3])
                    
                    # Vérifier les dimensions
                    if w <= 0 or h <= 0:
                        continue
                    
                    # Conversion en format YOLO (normalisé)
                    x_center = (x + w/2) / width
                    y_center = (y + h/2) / height
                    w_norm = w / width
                    h_norm = h / height
                    
                    # Vérifier que les coordonnées normalisées sont valides
                    if (x_center < 0 or x_center > 1 or 
                        y_center < 0 or y_center > 1 or 
                        w_norm <= 0 or w_norm > 1 or 
                        h_norm <= 0 or h_norm > 1):
                        continue
                    
                    # Ajouter les points de repère (landmarks) si disponibles
                    landmarks = []
                    if len(label) >= 19:  # Si nous avons des points de repère
                        for i in range(5):
                            lm_x = float(label[4 + i*3])
                            lm_y = float(label[5 + i*3])
                            visible = float(label[6 + i*3])  # 1=visible, 0=invisible, -1=absent
                            
                            # Normalisation
                            lm_x_norm = lm_x / width
                            lm_y_norm = lm_y / height
                            
                            # Si point non visible, utiliser -1
                            if visible != 1:
                                lm_x_norm = -1
                                lm_y_norm = -1
                            
                            landmarks.extend([lm_x_norm, lm_y_norm])
                    else:
                        # Si aucun point de repère, utiliser -1 pour tous
                        landmarks = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
                    
                    # Format de l'annotation YOLO avec landmarks
                    yolo_format = (
                        f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f} "
                        f"{landmarks[0]:.6f} {landmarks[1]:.6f} {landmarks[2]:.6f} {landmarks[3]:.6f} "
                        f"{landmarks[4]:.6f} {landmarks[5]:.6f} {landmarks[6]:.6f} {landmarks[7]:.6f} "
                        f"{landmarks[8]:.6f} {landmarks[9]:.6f}"
                    )
                    
                    valid_annotations.append(yolo_format)
                except Exception:
                    # Ignorer les annotations mal formatées
                    continue
            
            # Écrire les annotations valides
            with open(label_path, 'w') as f:
                if valid_annotations:
                    f.write('\n'.join(valid_annotations))
                    return True
                else:
                    # Fichier vide si aucune annotation valide
                    return False
            
        except Exception:
            return False
    
    def _filter_corrupted_images(self):
        """Filtre les images et annotations corrompues"""
        print("\n=== Filtrage des images et annotations corrompues ===")
        
        for set_name in ['train', 'val']:
            dataset_path = f'{self.data_dir}/{set_name}'
            labels_dir = f'{dataset_path}/labels'
            images_dir = f'{dataset_path}/images'
            
            if not os.path.exists(labels_dir):
                print(f"✗ Répertoire de labels non trouvé: {labels_dir}")
                continue
            
            corrupted_count = 0
            fixed_count = 0
            
            # Parcourir tous les fichiers d'étiquettes
            for root, _, files in os.walk(labels_dir):
                for file in files:
                    if file.endswith('.txt'):
                        label_path = os.path.join(root, file)
                        rel_path = os.path.relpath(label_path, labels_dir)
                        
                        # Vérifier l'image correspondante
                        image_name = os.path.splitext(rel_path)[0] + '.jpg'
                        image_path = os.path.join(images_dir, image_name)
                        
                        if not os.path.exists(image_path):
                            corrupted_count += 1
                            continue
                        
                        # Vérifier et corriger les annotations
                        try:
                            with open(label_path, 'r') as f:
                                lines = f.readlines()
                            
                            # Vérifier chaque ligne d'annotation
                            valid_lines = []
                            has_invalid = False
                            
                            for line in lines:
                                parts = line.strip().split()
                                
                                if len(parts) < 15:  # Format YOLOv5-face complet (1 classe + 4 bbox + 10 landmarks)
                                    has_invalid = True
                                    continue
                                
                                # Vérifier les coordonnées normalisées
                                try:
                                    class_id = int(parts[0])
                                    x, y, w, h = map(float, parts[1:5])
                                    
                                    # Valider les coordonnées
                                    if (x < 0 or x > 1 or y < 0 or y > 1 or 
                                        w <= 0 or w > 1 or h <= 0 or h > 1):
                                        has_invalid = True
                                        continue
                                    
                                    valid_lines.append(line)
                                except ValueError:
                                    has_invalid = True
                                    continue
                            
                            # Sauvegarder le fichier corrigé si nécessaire
                            if has_invalid:
                                with open(label_path, 'w') as f:
                                    f.writelines(valid_lines)
                                fixed_count += 1
                        
                        except Exception:
                            corrupted_count += 1
            
            print(f"✓ Filtrage terminé pour {set_name}: {corrupted_count} fichiers corrompus, {fixed_count} fichiers corrigés.")
    
    def _create_yaml_config(self):
        """Crée le fichier de configuration YAML pour l'entraînement"""
        print("\n=== Création du fichier de configuration YAML ===")
        
        yaml_content = {
            'train': f'{self.yolo_dir}/data/widerface/train',
            'val': f'{self.yolo_dir}/data/widerface/val',
            'test': f'{self.yolo_dir}/data/widerface/test',
            'nc': 1,
            'names': ['face'],
            'img_size': [640, 640],  # Taille par défaut
            'conf_thres': 0.25,
            'iou_thres': 0.45,
            'path': f'{self.yolo_dir}/data/widerface'
        }
        
        yaml_path = f'{self.yolo_dir}/data/widerface.yaml'
        
        # Créer le répertoire parent si nécessaire
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        
        # Écrire le fichier YAML
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
        
        print(f"✓ Fichier de configuration créé: {yaml_path}")
