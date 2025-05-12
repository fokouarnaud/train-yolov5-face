#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module pour l'évaluation et l'exportation du modèle YOLOv5-Face
"""

import os
import sys
import subprocess
import traceback
from pathlib import Path

# Importer la configuration centralisée
from config import DEFAULT_PATHS

class ModelEvaluator:
    """Classe pour gérer l'évaluation et l'exportation du modèle YOLOv5-Face"""
    
    def __init__(self, root_dir, yolo_dir, data_dir, img_size=640):
        """Initialise la classe d'évaluation du modèle"""
        self.root_dir = root_dir
        self.yolo_dir = yolo_dir
        self.data_dir = data_dir
        self.img_size = img_size
        
        # Chemins des modèles et résultats
        self.weights_path = f'{self.yolo_dir}/runs/train/face_detection_transfer/weights/best.pt'
        self.predictions_dir = f'{self.root_dir}/widerface_txt'
    
    def evaluate(self):
        """Évalue le modèle sur l'ensemble de validation"""
        print("\n=== Évaluation du modèle ===")
        
        if not os.path.exists(self.weights_path):
            print(f"✗ Modèle non trouvé: {self.weights_path}")
            return False
        
        try:
            # 1. Compiler l'outil d'évaluation WiderFace
            self._compile_evaluation_tool()
            
            # 2. Vérifier/créer le fichier wider_val.txt
            wider_val_path = self._get_wider_val_path()
            if not wider_val_path:
                print("✗ Impossible de trouver ou créer wider_val.txt")
                return False
            
            # 3. Exécuter test_widerface.py (qui gère à la fois la génération des prédictions et l'évaluation)
            success = self._run_test_widerface(wider_val_path)
            
            if success:
                print("\n✓ Évaluation terminée avec succès!")
                return True
            else:
                print("\n✗ Échec de l'évaluation")
                return False
            
        except Exception as e:
            print(f"✗ Erreur lors de l'évaluation: {e}")
            print(traceback.format_exc())
            return False
    
    def export(self):
        """Exporte le modèle au format ONNX"""
        print("\n=== Exportation du modèle ===")
        
        if not os.path.exists(self.weights_path):
            print(f"✗ Modèle non trouvé: {self.weights_path}")
            return False
        
        try:
            # Revenir au répertoire principal
            current_dir = os.getcwd()
            os.chdir(self.yolo_dir)
            
            # Exporter le modèle
            export_cmd = [
                sys.executable, 'export.py',
                '--weights', self.weights_path,
                '--img_size', str(self.img_size),
                '--batch_size', '1',
                '--dynamic'
            ]
            
            subprocess.run(export_cmd, check=True)
            
            # Revenir au répertoire d'origine
            os.chdir(current_dir)
            
            print("\n✓ Exportation terminée avec succès!")
            return True
            
        except Exception as e:
            print(f"✗ Erreur lors de l'exportation: {e}")
            print(traceback.format_exc())
            return False
    
    def _compile_evaluation_tool(self):
        """Compile l'outil d'évaluation WiderFace"""
        print("Compilation de l'outil d'évaluation...")
        
        eval_dir = os.path.join(self.yolo_dir, "widerface_evaluate")
        
        if not os.path.exists(eval_dir):
            print(f"✗ Répertoire d'évaluation non trouvé: {eval_dir}")
            raise FileNotFoundError(f"Répertoire d'évaluation non trouvé: {eval_dir}")
        
        # Sauvegarder le répertoire courant
        current_dir = os.getcwd()
        
        try:
            # Aller dans le répertoire d'évaluation
            os.chdir(eval_dir)
            
            # Compiler l'extension
            subprocess.run([sys.executable, 'setup.py', 'build_ext', '--inplace'], check=True)
            print("✓ Outil d'évaluation compilé")
            
            # Revenir au répertoire d'origine
            os.chdir(current_dir)
            return True
        except Exception as e:
            print(f"✗ Erreur lors de la compilation: {e}")
            
            # Revenir au répertoire d'origine
            os.chdir(current_dir)
            return False
    
    def _get_wider_val_path(self):
        """Trouve ou crée le fichier wider_val.txt"""
        print("Recherche du fichier wider_val.txt...")
        
        # Chemins possibles pour wider_val.txt
        possible_paths = [
            os.path.join(self.data_dir, "val", "wider_val.txt"),
            os.path.join(self.data_dir, "wider_val.txt"),
            os.path.join(self.yolo_dir, "data", "widerface", "val", "wider_val.txt")
        ]
        
        # Vérifier les chemins possibles
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✓ Fichier wider_val.txt trouvé: {path}")
                return path
        
        # Si le fichier n'existe pas, le créer
        print("Fichier wider_val.txt non trouvé, tentative de création...")
        
        val_images_dir = os.path.join(self.data_dir, "val", "images")
        if not os.path.exists(val_images_dir):
            print(f"✗ Répertoire des images de validation non trouvé: {val_images_dir}")
            return None
        
        # Créer le fichier wider_val.txt dans le premier emplacement possible
        new_wider_val_path = possible_paths[0]
        os.makedirs(os.path.dirname(new_wider_val_path), exist_ok=True)
        
        try:
            with open(new_wider_val_path, 'w') as f:
                events = [d for d in os.listdir(val_images_dir) if os.path.isdir(os.path.join(val_images_dir, d))]
                
                if events:
                    # Structure avec événements
                    for event in events:
                        event_dir = os.path.join(val_images_dir, event)
                        for file in sorted(os.listdir(event_dir)):
                            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                                f.write(f"{event}/{file}\n")
                else:
                    # Structure plate
                    for root, _, files in os.walk(val_images_dir):
                        for file in sorted(files):
                            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                                rel_path = os.path.relpath(os.path.join(root, file), val_images_dir)
                                f.write(f"{rel_path}\n")
            
            print(f"✓ Fichier wider_val.txt créé: {new_wider_val_path}")
            return new_wider_val_path
        except Exception as e:
            print(f"✗ Erreur lors de la création du fichier wider_val.txt: {e}")
            return None
    
    def _run_test_widerface(self, wider_val_path):
        """Exécute test_widerface.py pour l'évaluation"""
        print("Exécution de test_widerface.py pour l'évaluation...")
        
        # Créer le répertoire de prédictions
        os.makedirs(self.predictions_dir, exist_ok=True)
        
        # Sauvegarder le répertoire courant
        current_dir = os.getcwd()
        
        try:
            # Aller dans le répertoire yolov5-face
            os.chdir(self.yolo_dir)
            
            # Vérifier le chemin des images de validation
            val_images_dir = os.path.join(self.data_dir, "val", "images")
            if not os.path.exists(val_images_dir):
                print(f"✗ Répertoire des images de validation non trouvé: {val_images_dir}")
                return False
            
            # Exécuter test_widerface.py
            test_cmd = [
                sys.executable, 'test_widerface.py',
                '--weights', self.weights_path,
                '--img-size', str(self.img_size),
                '--conf-thres', '0.02',
                '--dataset_folder', val_images_dir,
                '--save_folder', self.predictions_dir,
                '--folder_pict', wider_val_path
            ]
            
            print(f"Commande: {' '.join(test_cmd)}")
            result = subprocess.run(test_cmd, capture_output=True, text=True)
            print(result.stdout)
            
            # Vérifier les résultats
            if "done." in result.stdout:
                # Exécuter l'évaluation WiderFace
                eval_dir = os.path.join(self.yolo_dir, "widerface_evaluate")
                gt_dir = os.path.join(eval_dir, "ground_truth")
                
                os.chdir(eval_dir)
                
                eval_cmd = [
                    sys.executable, "evaluation.py",
                    '--pred', self.predictions_dir,
                    '--gt', gt_dir
                ]
                
                eval_result = subprocess.run(eval_cmd, capture_output=True, text=True)
                print(eval_result.stdout)
                
                # Vérifier si tous les AP sont à 0.0
                if "Easy   Val AP: 0.0" in eval_result.stdout and "Medium Val AP: 0.0" in eval_result.stdout and "Hard   Val AP: 0.0" in eval_result.stdout:
                    print("⚠️ Tous les AP sont à 0.0, vérifiez vos modifications manuelles")
                    
                    # Instructions pour les modifications manuelles
                    print("\nModifications manuelles nécessaires:")
                    print("1. box_overlaps.pyx: Remplacer np.int_t par np.int64_t et np.int par np.int64")
                    print("2. test_widerface.py: Remplacer 'pred = model(img, augment=opt.augment)[0]' par:")
                    print("   'outputs = model(img, augment=opt.augment); pred = outputs[0] if isinstance(outputs, tuple) else outputs'")
                    print("3. evaluation.py: Ajouter des vérifications pour éviter les divisions par zéro dans dataset_pr_info")
                    
                    # Revenir au répertoire d'origine
                    os.chdir(current_dir)
                    return False
                
                # Revenir au répertoire d'origine
                os.chdir(current_dir)
                return True
            else:
                print("✗ test_widerface.py n'a pas terminé correctement")
                
                # Revenir au répertoire d'origine
                os.chdir(current_dir)
                return False
                
        except Exception as e:
            print(f"✗ Erreur lors de l'exécution de test_widerface.py: {e}")
            
            # Revenir au répertoire d'origine
            os.chdir(current_dir)
            return False
