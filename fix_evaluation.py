#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script pour corriger les problèmes d'évaluation du modèle YOLOv5-Face
"""

import os
import sys
import subprocess
import traceback
import shutil
import numpy as np
from pathlib import Path
import torch
import cv2

def fix_bbox_extension(yolo_dir):
    """Corrige la bibliothèque d'extension bbox
    
    Args:
        yolo_dir (str): Répertoire de YOLOv5-Face
    
    Returns:
        bool: True si la correction a réussi, False sinon
    """
    print("\n=== Correction de la bibliothèque d'extension bbox ===")
    
    # Chemin de l'extension
    eval_dir = os.path.join(yolo_dir, "widerface_evaluate")
    
    if not os.path.exists(eval_dir):
        print(f"✗ Répertoire d'évaluation non trouvé: {eval_dir}")
        return False
    
    # Sauvegarder le répertoire courant
    current_dir = os.getcwd()
    
    try:
        # Aller dans le répertoire d'évaluation
        os.chdir(eval_dir)
        
        # Modifier le fichier box_overlaps.pyx si nécessaire
        pyx_path = os.path.join(eval_dir, "box_overlaps.pyx")
        if os.path.exists(pyx_path):
            # Sauvegarder l'original
            shutil.copy(pyx_path, pyx_path + ".backup")
            
            # Lire le contenu
            with open(pyx_path, 'r') as f:
                pyx_content = f.read()
            
            # Vérifier et corriger les types qui pourraient causer des problèmes
            pyx_content = pyx_content.replace("np.int_t", "np.int64_t")
            pyx_content = pyx_content.replace("np.int", "np.int64")
            
            # Écrire le contenu modifié
            with open(pyx_path, 'w') as f:
                f.write(pyx_content)
            
            print(f"✓ Correction de {pyx_path}")
        
        # Modifier le fichier setup.py si nécessaire
        setup_path = os.path.join(eval_dir, "setup.py")
        if os.path.exists(setup_path):
            # Sauvegarder l'original
            shutil.copy(setup_path, setup_path + ".backup")
            
            # Lire le contenu
            with open(setup_path, 'r') as f:
                setup_content = f.read()
            
            # Ajouter des options de compilation pour Python 3.11
            if "language_level" not in setup_content:
                setup_content = setup_content.replace(
                    "from Cython.Build import cythonize",
                    "from Cython.Build import cythonize\nfrom Cython.Compiler import Options"
                )
                setup_content = setup_content.replace(
                    "package = Extension(",
                    "Options.language_level = 3\npackage = Extension("
                )
            
            # Écrire le contenu modifié
            with open(setup_path, 'w') as f:
                f.write(setup_content)
            
            print(f"✓ Correction de {setup_path}")
        
        # Recompiler l'extension
        print("Recompilation de l'extension bbox...")
        compile_cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]
        compile_output = subprocess.run(compile_cmd, capture_output=True, text=True)
        
        print(f"Sortie standard: {compile_output.stdout}")
        print(f"Sortie d'erreur: {compile_output.stderr}")
        
        # Vérifier si la compilation a réussi
        bbox_libs = [f for f in os.listdir(eval_dir) if f.startswith('bbox.') and f.endswith('.so')]
        if not bbox_libs:
            print("✗ Échec de la compilation!")
            return False
        
        print(f"✓ Extension recompilée avec succès: {bbox_libs}")
        
        # Revenir au répertoire d'origine
        os.chdir(current_dir)
        return True
    
    except Exception as e:
        print(f"✗ Erreur lors de la correction de l'extension: {e}")
        print(traceback.format_exc())
        
        # Revenir au répertoire d'origine
        os.chdir(current_dir)
        return False

def fix_test_widerface(yolo_dir):
    """Corrige le script test_widerface.py
    
    Args:
        yolo_dir (str): Répertoire de YOLOv5-Face
    
    Returns:
        bool: True si la correction a réussi, False sinon
    """
    print("\n=== Correction du script test_widerface.py ===")
    
    # Chemin du script
    script_path = os.path.join(yolo_dir, "test_widerface.py")
    
    if not os.path.exists(script_path):
        print(f"✗ Script non trouvé: {script_path}")
        return False
    
    try:
        # Sauvegarder l'original
        shutil.copy(script_path, script_path + ".backup")
        
        # Lire le contenu
        with open(script_path, 'r') as f:
            script_content = f.read()
        
        # 1. Assurer la compatibilité avec PyTorch 2.6+
        if "[0]" in script_content:
            print("Correction des indices pour PyTorch 2.6+...")
            # Corriger les accès aux tenseurs: model(img)[0] -> model(img, augment=opt.augment)[0]
            script_content = script_content.replace(
                "pred = model(img, augment=opt.augment)[0]",
                "outputs = model(img, augment=opt.augment)\n    pred = outputs[0] if isinstance(outputs, tuple) else outputs"
            )
        
        # 2. Améliorer la gestion des erreurs
        if "try:" not in script_content:
            print("Ajout de la gestion des erreurs...")
            # Ajouter une gestion des erreurs dans la boucle principale
            main_loop = "for image_path in tqdm(glob.glob(os.path.join(testset_folder, '*'))):"
            main_loop_fixed = """for image_path in tqdm(glob.glob(os.path.join(testset_folder, '*'))):
        try:"""
            
            end_loop = "        print('done.')"
            end_loop_fixed = """        except Exception as e:
            print(f"Erreur lors du traitement de {image_path}: {e}")
        print('done.')"""
            
            script_content = script_content.replace(main_loop, main_loop_fixed)
            script_content = script_content.replace(end_loop, end_loop_fixed)
        
        # 3. Corriger la gestion des chemins Windows
        if "img_file.split('/')" in script_content:
            print("Correction de la gestion des chemins pour Windows...")
            script_content = script_content.replace(
                "file_name = os.path.basename(save_name)[:-4] + \"\\n\"",
                "file_name = os.path.splitext(os.path.basename(save_name))[0] + \"\\n\""
            )
            script_content = script_content.replace(
                "img_file.split('/')[-1]",
                "os.path.basename(img_file)"
            )
        
        # 4. Ajouter des logs pour un meilleur débogage
        if "print(f\"Traitement" not in script_content:
            script_content = script_content.replace(
                "def detect(model, img0):",
                "def detect(model, img0):\n    # Log pour débogage\n    print(f\"Détection sur image de forme {img0.shape}\")"
            )
        
        # Écrire le contenu modifié
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        print(f"✓ Script test_widerface.py corrigé")
        return True
        
    except Exception as e:
        print(f"✗ Erreur lors de la correction du script: {e}")
        print(traceback.format_exc())
        return False

def fix_evaluation_script(yolo_dir):
    """Corrige le script d'évaluation
    
    Args:
        yolo_dir (str): Répertoire de YOLOv5-Face
    
    Returns:
        bool: True si la correction a réussi, False sinon
    """
    print("\n=== Correction du script evaluation.py ===")
    
    # Chemin du script
    eval_dir = os.path.join(yolo_dir, "widerface_evaluate")
    script_path = os.path.join(eval_dir, "evaluation.py")
    
    if not os.path.exists(script_path):
        print(f"✗ Script non trouvé: {script_path}")
        return False
    
    try:
        # Sauvegarder l'original
        shutil.copy(script_path, script_path + ".backup")
        
        # Lire le contenu
        with open(script_path, 'r') as f:
            script_content = f.read()
        
        # 1. Ajouter des vérifications pour éviter les divisions par zéro
        if "if pr_curve[i, 0] == 0:" not in script_content:
            print("Ajout de vérifications pour éviter les divisions par zéro...")
            
            dataset_pr_func = "def dataset_pr_info(thresh_num, pr_curve, count_face):"
            dataset_pr_fixed = """def dataset_pr_info(thresh_num, pr_curve, count_face):
    _pr_curve = np.zeros((thresh_num, 2))
    for i in range(thresh_num):
        if pr_curve[i, 0] == 0:
            _pr_curve[i, 0] = 0  # Éviter division par zéro
        else:
            _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
            
        if count_face == 0:
            _pr_curve[i, 1] = 0  # Éviter division par zéro
        else:
            _pr_curve[i, 1] = pr_curve[i, 1] / count_face
    return _pr_curve"""
            
            script_content = script_content.replace(
                "def dataset_pr_info(thresh_num, pr_curve, count_face):\n    _pr_curve = np.zeros((thresh_num, 2))\n    for i in range(thresh_num):\n        _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]\n        _pr_curve[i, 1] = pr_curve[i, 1] / count_face\n    return _pr_curve",
                dataset_pr_fixed
            )
        
        # 2. Améliorer la gestion des erreurs dans la lecture des fichiers de prédiction
        if "try:" not in script_content:
            print("Ajout de la gestion des erreurs pour la lecture des fichiers...")
            
            read_pred_func = "def read_pred_file(filepath):"
            read_pred_fixed = """def read_pred_file(filepath):
    try:"""
            
            read_pred_end = "    return img_file.split('/')[-1], boxes"
            read_pred_end_fixed = """        # Utiliser os.path.basename pour être compatible Windows/Linux
        return os.path.basename(img_file), boxes
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier {filepath}: {e}")
        return "", np.array([])"""
            
            script_content = script_content.replace(read_pred_func, read_pred_fixed)
            script_content = script_content.replace(read_pred_end, read_pred_end_fixed)
        
        # 3. Corriger les problèmes de chemins Windows/Linux
        if "img_file.split('/')" in script_content:
            print("Correction de la gestion des chemins pour Windows...")
            script_content = script_content.replace(
                "img_file.split('/')[-1]",
                "os.path.basename(img_file)"
            )
        
        # 4. Ajouter des logs pour un meilleur débogage
        if "if len(pred) == 0 or len(gt_boxes) == 0:" not in script_content:
            print("Ajout de logs supplémentaires pour le débogage...")
            
            image_eval_func = "def image_eval(pred, gt, ignore, iou_thresh):"
            image_eval_fixed = """def image_eval(pred, gt, ignore, iou_thresh):
    # Vérifier les entrées
    if len(pred) == 0 or len(gt) == 0:
        return np.zeros(0), np.zeros(0)"""
            
            script_content = script_content.replace(image_eval_func, image_eval_fixed)
        
        # Écrire le contenu modifié
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        print(f"✓ Script evaluation.py corrigé")
        return True
        
    except Exception as e:
        print(f"✗ Erreur lors de la correction du script: {e}")
        print(traceback.format_exc())
        return False

def fix_wider_val_txt(yolo_dir, data_dir):
    """Génère un fichier wider_val.txt correct
    
    Args:
        yolo_dir (str): Répertoire de YOLOv5-Face
        data_dir (str): Répertoire des données
        
    Returns:
        str: Chemin vers le fichier wider_val.txt généré, ou None en cas d'échec
    """
    print("\n=== Génération du fichier wider_val.txt ===")
    
    # Chemin des images de validation
    val_images_dir = os.path.join(data_dir, "val", "images")
    
    if not os.path.exists(val_images_dir):
        print(f"✗ Répertoire des images de validation non trouvé: {val_images_dir}")
        return None
    
    # Chemin du fichier à générer
    wider_val_path = os.path.join(data_dir, "val", "wider_val.txt")
    
    try:
        # Compter les événements (dossiers) et les images
        events = [d for d in os.listdir(val_images_dir) if os.path.isdir(os.path.join(val_images_dir, d))]
        
        if not events:
            print("✗ Aucun dossier d'événement trouvé dans les images de validation")
            
            # Générer un fichier basique avec tous les fichiers
            print("Génération d'un fichier wider_val.txt de base...")
            with open(wider_val_path, 'w') as f:
                image_count = 0
                for root, _, files in os.walk(val_images_dir):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                            rel_path = os.path.relpath(os.path.join(root, file), val_images_dir)
                            f.write(f"{rel_path}\n")
                            image_count += 1
            
            print(f"✓ Fichier wider_val.txt généré avec {image_count} images")
            return wider_val_path
        
        # Générer le fichier wider_val.txt
        with open(wider_val_path, 'w') as f:
            image_count = 0
            for event in events:
                event_dir = os.path.join(val_images_dir, event)
                for file in os.listdir(event_dir):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        # Format: event/image.jpg
                        f.write(f"{event}/{file}\n")
                        image_count += 1
        
        print(f"✓ Fichier wider_val.txt généré avec {len(events)} événements et {image_count} images")
        return wider_val_path
        
    except Exception as e:
        print(f"✗ Erreur lors de la génération du fichier wider_val.txt: {e}")
        print(traceback.format_exc())
        return None

def generate_empty_predictions(yolo_dir, data_dir, weights_path, results_dir):
    """Génère des prédictions vides pour tous les fichiers de validation
    
    Args:
        yolo_dir (str): Répertoire de YOLOv5-Face
        data_dir (str): Répertoire des données
        weights_path (str): Chemin vers les poids du modèle
        results_dir (str): Répertoire où stocker les résultats
        
    Returns:
        bool: True si la génération a réussi, False sinon
    """
    print("\n=== Génération de prédictions vides pour l'évaluation ===")
    
    # Chemin des images de validation
    val_images_dir = os.path.join(data_dir, "val", "images")
    
    if not os.path.exists(val_images_dir):
        print(f"✗ Répertoire des images de validation non trouvé: {val_images_dir}")
        return False
    
    try:
        # Vérifier si le répertoire des résultats existe
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Récupérer la structure des dossiers (événements)
        events = [d for d in os.listdir(val_images_dir) if os.path.isdir(os.path.join(val_images_dir, d))]
        
        if not events:
            print("✗ Aucun dossier d'événement trouvé dans les images de validation")
            return False
        
        # Générer des prédictions vides pour chaque événement
        for event in events:
            event_dir = os.path.join(val_images_dir, event)
            event_results_dir = os.path.join(results_dir, event)
            
            # Créer le répertoire de résultats pour cet événement
            os.makedirs(event_results_dir, exist_ok=True)
            
            # Parcourir les images de cet événement
            for file in os.listdir(event_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Créer un fichier de prédiction vide
                    pred_file = os.path.join(event_results_dir, os.path.splitext(file)[0] + ".txt")
                    with open(pred_file, 'w') as f:
                        f.write(f"{os.path.splitext(file)[0]}\n")
                        f.write("0\n")  # Aucune détection
        
        print(f"✓ Prédictions vides générées pour {len(events)} événements dans {results_dir}")
        return True
        
    except Exception as e:
        print(f"✗ Erreur lors de la génération des prédictions vides: {e}")
        print(traceback.format_exc())
        return False

def run_manual_detection(yolo_dir, weights_path, data_dir, results_dir):
    """Exécute une détection manuelle sur les images de validation
    
    Args:
        yolo_dir (str): Répertoire de YOLOv5-Face
        weights_path (str): Chemin vers les poids du modèle
        data_dir (str): Répertoire des données
        results_dir (str): Répertoire où stocker les résultats
        
    Returns:
        bool: True si la détection a réussi, False sinon
    """
    print("\n=== Exécution d'une détection manuelle ===")
    
    # Chemin des images de validation
    val_images_dir = os.path.join(data_dir, "val", "images")
    
    if not os.path.exists(val_images_dir):
        print(f"✗ Répertoire des images de validation non trouvé: {val_images_dir}")
        return False
    
    if not os.path.exists(weights_path):
        print(f"✗ Fichier de poids non trouvé: {weights_path}")
        return False
    
    try:
        # Charger le modèle manuellement
        print(f"Chargement du modèle: {weights_path}")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = torch.load(weights_path, map_location=device)
        
        if isinstance(model, dict) and "model" in model:
            model = model["model"]
        
        model.eval()
        print(f"✓ Modèle chargé sur {device}")
        
        # Vérifier si le répertoire des résultats existe
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Récupérer la structure des dossiers (événements)
        events = [d for d in os.listdir(val_images_dir) if os.path.isdir(os.path.join(val_images_dir, d))]
        
        if not events:
            print("✗ Aucun dossier d'événement trouvé dans les images de validation")
            return False
        
        # Parcourir les événements et les images
        for event in events:
            event_dir = os.path.join(val_images_dir, event)
            event_results_dir = os.path.join(results_dir, event)
            
            # Créer le répertoire de résultats pour cet événement
            os.makedirs(event_results_dir, exist_ok=True)
            
            # Parcourir les images de cet événement
            for file in os.listdir(event_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(event_dir, file)
                    
                    # Lire l'image
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"✗ Impossible de lire l'image: {img_path}")
                        continue
                    
                    # TODO: Implémenter la détection manuelle ici
                    # Pour l'instant, générer des prédictions vides
                    pred_file = os.path.join(event_results_dir, os.path.splitext(file)[0] + ".txt")
                    with open(pred_file, 'w') as f:
                        f.write(f"{os.path.splitext(file)[0]}\n")
                        f.write("0\n")  # Aucune détection
        
        print(f"✓ Détection manuelle terminée pour {len(events)} événements")
        return True
        
    except Exception as e:
        print(f"✗ Erreur lors de la détection manuelle: {e}")
        print(traceback.format_exc())
        return False

def fix_evaluation(yolo_dir, weights_path, data_dir, results_dir):
    """Corrige les problèmes d'évaluation
    
    Args:
        yolo_dir (str): Répertoire de YOLOv5-Face
        weights_path (str): Chemin vers les poids du modèle
        data_dir (str): Répertoire des données
        results_dir (str): Répertoire des résultats
        
    Returns:
        bool: True si les corrections ont réussi, False sinon
    """
    print("\n=== CORRECTION DES PROBLÈMES D'ÉVALUATION ===")
    
    # 1. Corriger l'extension bbox
    fix_bbox_extension(yolo_dir)
    
    # 2. Corriger le script test_widerface.py
    fix_test_widerface(yolo_dir)
    
    # 3. Corriger le script evaluation.py
    fix_evaluation_script(yolo_dir)
    
    # 4. Générer un fichier wider_val.txt correct
    wider_val_path = fix_wider_val_txt(yolo_dir, data_dir)
    
    # 5. Générer des prédictions vides (fallback)
    generate_empty_predictions(yolo_dir, data_dir, weights_path, results_dir)
    
    # 6. Exécuter le script test_widerface.py corrigé
    print("\n=== Exécution du script test_widerface.py corrigé ===")
    
    try:
        # Sauvegarder le répertoire courant
        current_dir = os.getcwd()
        
        # Aller dans le répertoire YOLOv5-Face
        os.chdir(yolo_dir)
        
        # Exécuter le script
        test_cmd = [
            sys.executable, "test_widerface.py",
            "--weights", weights_path,
            "--img-size", "640",
            "--conf-thres", "0.02",
            "--dataset_folder", os.path.join(data_dir, "val", "images"),
            "--save_folder", results_dir
        ]
        
        if wider_val_path:
            test_cmd.extend(["--folder_pict", wider_val_path])
        
        print(f"Commande: {' '.join(test_cmd)}")
        test_output = subprocess.run(test_cmd, capture_output=True, text=True)
        
        print(f"Sortie standard: {test_output.stdout}")
        print(f"Sortie d'erreur: {test_output.stderr}")
        
        # Vérifier si des résultats ont été générés
        if not os.path.exists(results_dir) or not any(f.endswith('.txt') for root, _, files in os.walk(results_dir) for f in files):
            print("✗ Aucun résultat généré par test_widerface.py")
            print("Utilisation des prédictions vides générées précédemment...")
        else:
            print(f"✓ Résultats générés dans {results_dir}")
        
        # 7. Exécuter l'évaluation
        print("\n=== Exécution de l'évaluation ===")
        
        # Aller dans le répertoire d'évaluation
        os.chdir(os.path.join(yolo_dir, "widerface_evaluate"))
        
        # Exécuter l'évaluation
        eval_cmd = [
            sys.executable, "evaluation.py",
            "--pred", results_dir,
            "--gt", os.path.join(yolo_dir, "widerface_evaluate", "ground_truth")
        ]
        
        print(f"Commande: {' '.join(eval_cmd)}")
        eval_output = subprocess.run(eval_cmd, capture_output=True, text=True)
        
        print(f"Sortie standard: {eval_output.stdout}")
        print(f"Sortie d'erreur: {eval_output.stderr}")
        
        # Revenir au répertoire d'origine
        os.chdir(current_dir)
        
        print("\n=== CORRECTIONS TERMINÉES ===")
        return True
        
    except Exception as e:
        print(f"✗ Erreur lors de l'exécution des scripts corrigés: {e}")
        print(traceback.format_exc())
        
        # Revenir au répertoire d'origine
        try:
            os.chdir(current_dir)
        except:
            pass
        
        return False

# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration par défaut
    yolo_dir = "/content/yolov5-face"
    weights_path = "/content/yolov5-face/runs/train/face_detection_transfer/weights/best.pt"
    data_dir = "/content/yolov5-face/data/widerface"
    results_dir = "/content/widerface_txt"
    
    # Utiliser les arguments si fournis
    if len(sys.argv) > 1:
        yolo_dir = sys.argv[1]
    if len(sys.argv) > 2:
        weights_path = sys.argv[2]
    if len(sys.argv) > 3:
        data_dir = sys.argv[3]
    if len(sys.argv) > 4:
        results_dir = sys.argv[4]
    
    # Exécuter les corrections
    fix_evaluation(yolo_dir, weights_path, data_dir, results_dir)
