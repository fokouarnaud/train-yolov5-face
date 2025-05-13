#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module pour l'évaluation et l'exportation du modèle YOLOv5-Face
"""

import os
import sys
import subprocess
import traceback
import glob
import shutil
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
        
        # Répertoire de débogage
        self.debug_dir = f'{self.root_dir}/debug_output'
        os.makedirs(self.debug_dir, exist_ok=True)
    
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
                self._collect_debug_info()
                return False
            
        except Exception as e:
            print(f"✗ Erreur lors de l'évaluation: {e}")
            print(traceback.format_exc())
            self._collect_debug_info()
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
                    # Structure avec événements (attendue pour WiderFace)
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
                
            # Préparer la structure de répertoires pour les événements
            try:
                # Lire les événements du fichier wider_val.txt
                event_dirs = set()
                with open(wider_val_path, 'r') as f:
                    for line in f:
                        parts = line.strip().replace('\\', '/').split('/')
                        if len(parts) >= 2:
                            event_dirs.add(parts[0])
                
                # Créer les répertoires d'événements à l'avance
                for event in event_dirs:
                    os.makedirs(os.path.join(self.predictions_dir, event), exist_ok=True)
                    
                print(f"Répertoires préparés pour {len(event_dirs)} événements")
            except Exception as e:
                print(f"Avertissement: Impossible de préparer les répertoires d'événements: {e}")
            
            # Exécuter test_widerface.py avec un seuil de confiance plus bas
            test_cmd = [
                sys.executable, 'test_widerface.py',
                '--weights', self.weights_path,
                '--img-size', str(self.img_size),
                '--conf-thres', '0.01',  # Réduit pour détecter plus de visages
                '--dataset_folder', val_images_dir,
                '--save_folder', self.predictions_dir,
                '--folder_pict', wider_val_path
            ]
            
            print(f"Commande: {' '.join(test_cmd)}")
            result = subprocess.run(test_cmd, capture_output=True, text=True)
            print(result.stdout)
            
            # Sauvegarder la sortie pour débogage
            with open(os.path.join(self.debug_dir, "test_widerface_output.log"), 'w') as f:
                f.write(f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")
            
            # Vérifier les résultats
            if "done." in result.stdout:
                # Vérifier le contenu du répertoire de prédictions
                print("Vérification des fichiers de prédiction générés...")
                pred_events = [d for d in os.listdir(self.predictions_dir) 
                             if os.path.isdir(os.path.join(self.predictions_dir, d))]
                
                if not pred_events:
                    print("⚠️ Aucun répertoire d'événement trouvé dans les prédictions!")
                    os.chdir(current_dir)
                    return False
                    
                print(f"Événements générés ({len(pred_events)}): {pred_events}")
                
                total_files = 0
                for event in pred_events:
                    files = os.listdir(os.path.join(self.predictions_dir, event))
                    total_files += len(files)
                    print(f"  - {event}: {len(files)} fichiers")
                    
                print(f"Total de fichiers de prédiction: {total_files}")
                
                if total_files == 0:
                    print("⚠️ Aucun fichier de prédiction généré!")
                    os.chdir(current_dir)
                    return False
                
                # Exécuter l'évaluation WiderFace
                eval_dir = os.path.join(self.yolo_dir, "widerface_evaluate")
                gt_dir = os.path.join(eval_dir, "ground_truth")
                
                os.chdir(eval_dir)
                
                eval_cmd = [
                    sys.executable, "evaluation.py",
                    '--pred', self.predictions_dir,
                    '--gt', gt_dir
                ]
                
                print("Exécution de l'évaluation...")
                eval_result = subprocess.run(eval_cmd, capture_output=True, text=True)
                print(eval_result.stdout)
                
                # Sauvegarder la sortie pour débogage
                with open(os.path.join(self.debug_dir, "evaluation_output.log"), 'w') as f:
                    f.write(f"STDOUT:\n{eval_result.stdout}\n\nSTDERR:\n{eval_result.stderr}")
                
                # Vérifier si tous les AP sont à 0.0
                if "Easy   Val AP: 0.0" in eval_result.stdout and "Medium Val AP: 0.0" in eval_result.stdout and "Hard   Val AP: 0.0" in eval_result.stdout:
                    print("⚠️ Tous les AP sont à 0.0, vérifiez vos modifications manuelles")
                    
                    # Instructions pour les modifications manuelles
                    print("\nModifications manuelles nécessaires:")
                    print("1. box_overlaps.pyx: Remplacer np.int_t par np.int64_t et np.int par np.int64")
                    print("2. test_widerface.py: Remplacer 'pred = model(img, augment=opt.augment)[0]' par:")
                    print("   'outputs = model(img, augment=opt.augment); pred = outputs[0] if isinstance(outputs, tuple) else outputs'")
                    print("3. evaluation.py: Ajouter des vérifications pour éviter les divisions par zéro dans dataset_pr_info")
                    print("4. test_widerface.py: Assurez-vous que glob.glob utilise os.path.join(testset_folder, '*', '*') pour trouver les images")
                    print("5. test_widerface.py: Réduisez le seuil de confiance à 0.01")
                    
                    # Collecter des informations de débogage
                    self._collect_debug_info()
                    
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
            print(traceback.format_exc())
            
            # Revenir au répertoire d'origine
            os.chdir(current_dir)
            return False
    
    def _collect_debug_info(self):
        """Collecte des informations de débogage pour aider à résoudre les problèmes"""
        try:
            print("Collecte d'informations de débogage...")
            
            # Vérifier la structure du répertoire des images de validation
            val_images_dir = os.path.join(self.data_dir, "val", "images")
            if os.path.exists(val_images_dir):
                with open(os.path.join(self.debug_dir, "val_images_structure.log"), 'w') as f:
                    f.write(f"Structure du répertoire des images de validation: {val_images_dir}\n\n")
                    total_files = 0
                    
                    # Parcourir la structure de répertoires
                    for root, dirs, files in os.walk(val_images_dir):
                        level = root.replace(val_images_dir, '').count(os.sep)
                        indent = ' ' * 4 * level
                        f.write(f"{indent}{os.path.basename(root)}/\n")
                        
                        for file in files:
                            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                                total_files += 1
                                if total_files <= 20:  # Limiter l'affichage pour éviter les fichiers trop volumineux
                                    f.write(f"{indent}    {file}\n")
                    
                    f.write(f"\nTotal des fichiers image: {total_files}\n")
            
            # Enregistrer des informations sur les prédictions générées
            with open(os.path.join(self.debug_dir, "predictions.log"), 'w') as f:
                f.write(f"Répertoire de prédictions: {self.predictions_dir}\n\n")
                
                if os.path.exists(self.predictions_dir):
                    events = [d for d in os.listdir(self.predictions_dir) 
                             if os.path.isdir(os.path.join(self.predictions_dir, d))]
                    
                    f.write(f"Événements trouvés ({len(events)}): {events}\n\n")
                    
                    for event in events:
                        event_dir = os.path.join(self.predictions_dir, event)
                        files = os.listdir(event_dir)
                        f.write(f"Événement '{event}': {len(files)} fichiers\n")
                        
                        # Echantillon des fichiers
                        for file in files[:5]:  # Afficher les 5 premiers fichiers
                            file_path = os.path.join(event_dir, file)
                            try:
                                with open(file_path, 'r') as txt_file:
                                    content = txt_file.read()
                                    f.write(f"  - {file}:\n{content}\n")
                            except Exception as e:
                                f.write(f"  - {file}: Erreur de lecture: {e}\n")
                else:
                    f.write("Le répertoire de prédictions n'existe pas!\n")
            
            # Vérifier les modifications apportées aux fichiers
            with open(os.path.join(self.debug_dir, "modifications.log"), 'w') as f:
                f.write("Vérification des modifications apportées aux fichiers source\n\n")
                
                # Vérifier test_widerface.py
                test_widerface_path = os.path.join(self.yolo_dir, "test_widerface.py")
                if os.path.exists(test_widerface_path):
                    f.write(f"Analyse de test_widerface.py:\n")
                    with open(test_widerface_path, 'r') as src_file:
                        content = src_file.read()
                        
                        # Vérifier la structure du parcours des images
                        if "glob.glob(os.path.join(testset_folder, '*', '*'))" in content:
                            f.write("  - ✓ Utilise glob.glob(os.path.join(testset_folder, '*', '*')) pour trouver les images\n")
                        else:
                            f.write("  - ✗ N'utilise PAS glob.glob(os.path.join(testset_folder, '*', '*')) pour trouver les images\n")
                        
                        # Vérifier la gestion des sorties PyTorch
                        if "if isinstance(output, tuple)" in content:
                            f.write("  - ✓ Contient une gestion pour différents formats de sortie PyTorch\n")
                        else:
                            f.write("  - ✗ Ne contient PAS de gestion pour différents formats de sortie PyTorch\n")
                        
                        # Vérifier le seuil de confiance par défaut
                        if "default=0.01" in content and "--conf-thres" in content:
                            f.write("  - ✓ Utilise un seuil de confiance par défaut de 0.01\n")
                        else:
                            f.write("  - ✗ N'utilise PAS un seuil de confiance par défaut de 0.01\n")
                
                # Vérifier evaluation.py
                evaluation_path = os.path.join(self.yolo_dir, "widerface_evaluate", "evaluation.py")
                if os.path.exists(evaluation_path):
                    f.write(f"\nAnalyse de evaluation.py:\n")
                    with open(evaluation_path, 'r') as src_file:
                        content = src_file.read()
                        
                        # Vérifier la protection contre la division par zéro
                        if "if pr_curve[i, 0] > 0" in content and "dataset_pr_info" in content:
                            f.write("  - ✓ Contient une protection contre la division par zéro dans dataset_pr_info\n")
                        else:
                            f.write("  - ✗ Ne contient PAS de protection contre la division par zéro dans dataset_pr_info\n")
                        
                        # Vérifier le traitement des événements manquants
                        if "Warning: Event" in content and "not found in predictions" in content:
                            f.write("  - ✓ Contient un avertissement pour les événements manquants\n")
                        else:
                            f.write("  - ✗ Ne contient PAS d'avertissement pour les événements manquants\n")
                
                # Vérifier box_overlaps.pyx
                box_overlaps_path = os.path.join(self.yolo_dir, "widerface_evaluate", "box_overlaps.pyx")
                if os.path.exists(box_overlaps_path):
                    f.write(f"\nAnalyse de box_overlaps.pyx:\n")
                    with open(box_overlaps_path, 'r') as src_file:
                        content = src_file.read()
                        
                        # Vérifier les types np.int vs np.int64
                        if "np.int_t" in content:
                            f.write("  - ✗ Contient encore np.int_t qui doit être remplacé par np.int64_t\n")
                        else:
                            f.write("  - ✓ Ne contient PAS np.int_t\n")
                        
                        if "np.int" in content and "np.int64" not in content and "np.int32" not in content:
                            f.write("  - ✗ Contient encore np.int qui doit être remplacé par np.int64\n")
                        else:
                            f.write("  - ✓ Ne contient PAS np.int ou a été correctement remplacé\n")
            
            print(f"Informations de débogage enregistrées dans {self.debug_dir}")
            return True
        except Exception as e:
            print(f"Erreur lors de la collecte des informations de débogage: {e}")
            print(traceback.format_exc())
            return False
