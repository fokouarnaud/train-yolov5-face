#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module pour l'évaluation et l'exportation du modèle YOLOv5-Face
"""

import os
import subprocess
import traceback
from pathlib import Path

# Importer la configuration centralisée
from config import DEFAULT_PATHS

class ModelEvaluator:
    """Classe pour gérer l'évaluation et l'exportation du modèle YOLOv5-Face"""
    
    def __init__(self, root_dir, yolo_dir, data_dir, img_size=640):
        """Initialise la classe d'évaluation du modèle
        
        Args:
            root_dir (str): Répertoire racine de Colab
            yolo_dir (str): Répertoire de YOLOv5-Face
            data_dir (str): Répertoire des données
            img_size (int): Taille d'image pour l'évaluation
        """
        self.root_dir = root_dir
        self.yolo_dir = yolo_dir
        self.data_dir = data_dir
        self.img_size = img_size
        
        # Chemins des modèles et résultats
        self.weights_path = f'{self.yolo_dir}/runs/train/face_detection_transfer/weights/best.pt'
        self.predictions_dir = f'{self.root_dir}/widerface_txt'
        self.results_dir = f'{self.root_dir}/widerface_results'
    
    def evaluate(self):
        """Évalue le modèle sur l'ensemble de validation
        
        Returns:
            bool: True si l'évaluation a réussi, False sinon
        """
        print("\n=== Évaluation du modèle ===")
        
        # Vérifier si le modèle existe
        if not os.path.exists(self.weights_path):
            print(f"✗ Modèle non trouvé: {self.weights_path}")
            return False
        
        try:
            # Créer un répertoire pour les prédictions
            os.makedirs(self.predictions_dir, exist_ok=True)
            
            # Exécuter la détection sur l'ensemble de validation
            self._run_detection()
            
            # Compiler l'outil d'évaluation
            self._compile_evaluation_tool()
            
            # Formater les prédictions
            self._format_predictions()
            
            # Exécuter l'évaluation
            self._run_evaluation()
            
            print("\n✓ Évaluation terminée avec succès!")
            return True
            
        except Exception as e:
            print(f"✗ Erreur lors de l'évaluation: {e}")
            print(traceback.format_exc())
            return False
    
    def export(self):
        """Exporte le modèle au format ONNX
        
        Returns:
            bool: True si l'exportation a réussi, False sinon
        """
        print("\n=== Exportation du modèle ===")
        
        # Vérifier si le modèle existe
        if not os.path.exists(self.weights_path):
            print(f"✗ Modèle non trouvé: {self.weights_path}")
            return False
        
        try:
            # Revenir au répertoire principal
            os.chdir(self.yolo_dir)
            
            # Exporter le modèle
            export_cmd = [
                'python', f'{self.yolo_dir}/export.py',
                '--weights', self.weights_path,
                '--img_size', str(self.img_size),
                '--batch_size', '1',
                '--dynamic'
            ]
            
            subprocess.run(export_cmd, check=True)
            print("\n✓ Exportation terminée avec succès!")
            return True
            
        except Exception as e:
            print(f"✗ Erreur lors de l'exportation: {e}")
            print(traceback.format_exc())
            return False
    
    def _run_detection(self):
        """Exécute la détection sur l'ensemble de validation"""
        print("Exécution de la détection sur l'ensemble de validation...")
        
        detect_cmd = [
            'python', f'{self.yolo_dir}/detect_face.py',
            '--weights', self.weights_path,
            '--source', f'{self.data_dir}/val/images',
            '--img', str(self.img_size),
            '--conf', '0.001',
            '--save-txt',
            '--save-conf',
            '--output', self.predictions_dir
        ]
        
        subprocess.run(detect_cmd, check=True)
        print("✓ Détection terminée sur l'ensemble de validation")
    
    def _compile_evaluation_tool(self):
        """Compile l'outil d'évaluation WiderFace"""
        print("Compilation de l'outil d'évaluation...")
        
        eval_dir = f'{self.yolo_dir}/widerface_evaluate'
        
        if not os.path.exists(eval_dir):
            print(f"✗ Répertoire d'évaluation non trouvé: {eval_dir}")
            raise FileNotFoundError(f"Répertoire d'évaluation non trouvé: {eval_dir}")
        
        # Aller dans le répertoire d'évaluation
        os.chdir(eval_dir)
        
        # Compiler l'outil d'évaluation
        subprocess.run(['python', 'setup.py', 'build_ext', '--inplace'], check=True)
        print("✓ Outil d'évaluation compilé")
    
    def _format_predictions(self):
        """Formate les prédictions pour l'évaluation WiderFace"""
        print("Formatage des prédictions pour l'évaluation...")
        
        import cv2
        import glob
        
        # Créer le répertoire de sortie
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Parcourir les résultats et les organiser selon la structure attendue
        for txt_file in glob.glob(f"{self.predictions_dir}/*.txt"):
            # Obtenir le nom de fichier de base
            basename = os.path.basename(txt_file)
            image_name = basename.replace('.txt', '.jpg')
            
            # Déterminer le nom de l'événement (répertoire parent)
            with open(txt_file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) > 0:
                try:
                    # Extraire le nom de l'événement (première ligne contient le chemin)
                    path_line = lines[0].strip()
                    if '/' in path_line:
                        event_name = path_line.split('/')[-2]  # Format: .../event_name/image_name
                    else:
                        # Si le format est différent, utiliser un événement par défaut
                        event_name = 'unknown'
                    
                    # Créer le répertoire de l'événement s'il n'existe pas
                    event_dir = os.path.join(self.results_dir, event_name)
                    os.makedirs(event_dir, exist_ok=True)
                    
                    # Créer le fichier de sortie au format attendu
                    out_file = os.path.join(event_dir, image_name.replace('.jpg', '.txt'))
                    
                    with open(out_file, 'w') as f:
                        f.write(f"{image_name}\n")
                        f.write(f"{len(lines) - 1}\n")  # Nombre de détections (sans la ligne de chemin)
                        
                        # Lire l'image pour obtenir ses dimensions
                        img_path = os.path.join(f'{self.data_dir}/val/images', image_name)
                        if os.path.exists(img_path):
                            img = cv2.imread(img_path)
                            if img is not None:
                                img_height, img_width = img.shape[:2]
                                
                                for line in lines[1:]:  # Ignorer la première ligne (chemin)
                                    parts = line.strip().split()
                                    if len(parts) >= 6:  # Format YOLO: class x_center y_center width height conf
                                        class_id, x_center, y_center, width, height, conf = parts[:6]
                                        
                                        # Convertir du format YOLO (normalisé) au format WiderFace (pixels)
                                        x_center, y_center, width, height = map(float, [x_center, y_center, width, height])
                                        conf = float(conf)
                                        
                                        # Calculer les coordonnées en pixels
                                        x1 = (x_center - width/2) * img_width
                                        y1 = (y_center - height/2) * img_height
                                        w = width * img_width
                                        h = height * img_height
                                        
                                        # Écrire au format WiderFace
                                        f.write(f"{x1:.1f} {y1:.1f} {w:.1f} {h:.1f} {conf:.6f}\n")
                
                except Exception as e:
                    print(f"✗ Erreur lors du formatage de {txt_file}: {e}")
        
        print(f"✓ Prédictions formatées et enregistrées dans {self.results_dir}")
    
    def _run_evaluation(self):
        """Exécute l'évaluation WiderFace"""
        print("Exécution de l'évaluation...")
        
        # Vérifier si le répertoire de prédictions existe et n'est pas vide
        if not os.path.exists(self.results_dir) or not os.listdir(self.results_dir):
            print(f"✗ Répertoire de prédictions vide ou non trouvé: {self.results_dir}")
            raise FileNotFoundError(f"Répertoire de prédictions vide ou non trouvé: {self.results_dir}")
        
        # Vérifier si le répertoire de ground truth existe
        gt_dir = f'{self.yolo_dir}/widerface_evaluate/ground_truth'
        if not os.path.exists(gt_dir):
            print(f"✗ Répertoire de ground truth non trouvé: {gt_dir}")
            # On continue quand même, l'évaluation s'adaptera
        
        # Exécuter l'évaluation
        eval_cmd = [
            'python', f'{self.yolo_dir}/widerface_evaluate/evaluation.py',
            '--pred', self.results_dir,
            '--gt', gt_dir
        ]
        
        subprocess.run(eval_cmd, check=True)
        print("✓ Évaluation WiderFace terminée")
