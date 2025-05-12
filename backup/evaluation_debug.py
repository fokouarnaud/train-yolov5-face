#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script pour déboguer l'évaluation du modèle YOLOv5-Face
"""

import os
import sys
import subprocess
import traceback
from pathlib import Path
import numpy as np
import torch
import shutil

def debug_evaluation(yolo_dir, weights_path, data_dir, results_dir):
    """Débogue l'évaluation du modèle
    
    Args:
        yolo_dir (str): Répertoire de YOLOv5-Face
        weights_path (str): Chemin vers les poids du modèle
        data_dir (str): Répertoire des données
        results_dir (str): Répertoire des résultats
    """
    print("\n=== DÉBOGAGE DE L'ÉVALUATION ===")
    
    # Étape 1: Vérifier l'existence des fichiers et des répertoires
    print("\n1. Vérification des fichiers et répertoires")
    if not os.path.exists(weights_path):
        print(f"✗ ERREUR: Le fichier de poids n'existe pas: {weights_path}")
        return
    else:
        print(f"✓ Fichier de poids trouvé: {weights_path}")
        # Afficher la taille du fichier
        file_size = os.path.getsize(weights_path) / (1024 * 1024)  # En Mo
        print(f"   Taille du fichier: {file_size:.2f} Mo")
    
    val_images_dir = f'{data_dir}/val/images'
    if not os.path.exists(val_images_dir):
        print(f"✗ ERREUR: Le répertoire des images de validation n'existe pas: {val_images_dir}")
        # Vérifier une alternative
        print("Cherche des répertoires d'images alternatifs...")
        for parent_dir in [data_dir, os.path.dirname(data_dir)]:
            for root, dirs, _ in os.walk(parent_dir):
                for dir_name in dirs:
                    if 'val' in dir_name.lower() and 'image' in dir_name.lower():
                        alt_dir = os.path.join(root, dir_name)
                        print(f"Alternative possible: {alt_dir}")
        return
    else:
        print(f"✓ Répertoire des images de validation trouvé: {val_images_dir}")
        # Compter les images
        image_count = sum(len(files) for _, _, files in os.walk(val_images_dir) 
                        if any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in files))
        print(f"   Nombre d'images trouvées: {image_count}")
    
    # Étape 2: Vérifier l'outil d'évaluation
    print("\n2. Vérification de l'outil d'évaluation")
    eval_dir = f'{yolo_dir}/widerface_evaluate'
    if not os.path.exists(eval_dir):
        print(f"✗ ERREUR: Le répertoire d'évaluation n'existe pas: {eval_dir}")
        return
    else:
        print(f"✓ Répertoire d'évaluation trouvé: {eval_dir}")
    
    # Vérifier les fichiers de vérité terrain
    gt_dir = f'{eval_dir}/ground_truth'
    if not os.path.exists(gt_dir):
        print(f"✗ ERREUR: Le répertoire de vérité terrain n'existe pas: {gt_dir}")
        return
    else:
        print(f"✓ Répertoire de vérité terrain trouvé: {gt_dir}")
        # Lister les fichiers
        gt_files = os.listdir(gt_dir)
        for gt_file in gt_files:
            print(f"   - {gt_file} ({os.path.getsize(os.path.join(gt_dir, gt_file)) / 1024:.2f} Ko)")
    
    # Étape 3: Vérifier les bibliothèques d'extension
    print("\n3. Vérification des bibliothèques d'extension")
    bbox_libs = [f for f in os.listdir(eval_dir) if f.startswith('bbox.') and f.endswith('.so')]
    if not bbox_libs:
        print(f"✗ ERREUR: Bibliothèque bbox non trouvée dans {eval_dir}")
        print("   La compilation de l'outil d'évaluation a peut-être échoué.")
        
        # Essayer de recompiler
        print("   Tentative de recompilation...")
        os.chdir(eval_dir)
        setup_output = subprocess.run(['python', 'setup.py', 'build_ext', '--inplace'], 
                                       capture_output=True, text=True)
        print(f"Sortie standard: {setup_output.stdout}")
        print(f"Sortie d'erreur: {setup_output.stderr}")
        
        # Vérifier à nouveau
        bbox_libs = [f for f in os.listdir(eval_dir) if f.startswith('bbox.') and f.endswith('.so')]
        if not bbox_libs:
            print("✗ ERREUR: Recompilation échouée!")
            return
    else:
        print(f"✓ Bibliothèques bbox trouvées: {bbox_libs}")
    
    # Étape 4: Tenter une détection sur quelques images
    print("\n4. Test de détection sur quelques images")
    try:
        # Créer un répertoire temporaire pour les résultats de test
        test_dir = os.path.join(yolo_dir, "test_detection")
        os.makedirs(test_dir, exist_ok=True)
        
        # Lister quelques images de validation (max 5)
        test_images = []
        for root, _, files in os.walk(val_images_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    test_images.append(os.path.join(root, file))
                    if len(test_images) >= 5:
                        break
            if len(test_images) >= 5:
                break
        
        if not test_images:
            print("✗ ERREUR: Aucune image de test trouvée!")
            return
        
        print(f"   Nombre d'images de test: {len(test_images)}")
        for img_path in test_images:
            print(f"   - {os.path.basename(img_path)}")
        
        # Exécuter la détection
        print("\n   Exécution de la détection...")
        detect_cmd = [
            'python', f'{yolo_dir}/detect_face.py',
            '--weights', weights_path,
            '--source', test_images[0],  # Utiliser la première image
            '--img-size', '640',
            '--conf-thres', '0.02',
            '--save-txt',
            '--project', test_dir,
            '--name', 'test_detection',
            '--exist-ok'
        ]
        
        detect_output = subprocess.run(detect_cmd, capture_output=True, text=True)
        print(f"   Sortie standard: {detect_output.stdout}")
        print(f"   Sortie d'erreur: {detect_output.stderr}")
        
        # Vérifier si des résultats ont été générés
        detection_results = os.path.join(test_dir, "test_detection", "labels")
        if os.path.exists(detection_results) and os.listdir(detection_results):
            print(f"✓ Détection réussie! Résultats dans {detection_results}")
            for result_file in os.listdir(detection_results):
                print(f"   - {result_file}")
                with open(os.path.join(detection_results, result_file), 'r') as f:
                    content = f.read()
                    print(f"     Contenu: {content[:100]}..." if len(content) > 100 else f"     Contenu: {content}")
        else:
            print(f"✗ ERREUR: Aucun résultat de détection n'a été généré!")
    except Exception as e:
        print(f"✗ ERREUR lors du test de détection: {e}")
        print(traceback.format_exc())
    
    # Étape 5: Vérifier le script test_widerface.py
    print("\n5. Vérification du script test_widerface.py")
    test_widerface_path = os.path.join(yolo_dir, "test_widerface.py")
    if not os.path.exists(test_widerface_path):
        print(f"✗ ERREUR: Le script test_widerface.py n'existe pas: {test_widerface_path}")
        return
    else:
        print(f"✓ Script test_widerface.py trouvé: {test_widerface_path}")
    
    # Étape 6: Vérifier les prédictions existantes
    print("\n6. Vérification des prédictions existantes")
    if not os.path.exists(results_dir):
        print(f"✗ ERREUR: Le répertoire des résultats n'existe pas: {results_dir}")
        print("   Une exécution précédente de test_widerface.py a peut-être échoué.")
    else:
        print(f"✓ Répertoire des résultats trouvé: {results_dir}")
        # Compter les fichiers de prédiction
        txt_count = 0
        for root, _, files in os.walk(results_dir):
            for file in files:
                if file.endswith('.txt'):
                    txt_count += 1
        
        print(f"   Nombre de fichiers de prédiction: {txt_count}")
        
        # Examiner quelques fichiers de prédiction (max 5)
        print("   Échantillon de fichiers de prédiction:")
        sample_count = 0
        for root, _, files in os.walk(results_dir):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                            print(f"   - {os.path.relpath(file_path, results_dir)}: {content[:100]}..." if len(content) > 100 else f"   - {os.path.relpath(file_path, results_dir)}: {content}")
                            sample_count += 1
                            if sample_count >= 5:
                                break
                    except Exception as e:
                        print(f"   - {os.path.relpath(file_path, results_dir)}: ERREUR DE LECTURE: {e}")
            if sample_count >= 5:
                break
    
    # Étape 7: Vérifier la compatibilité Python/PyTorch
    print("\n7. Vérification de la compatibilité Python/PyTorch")
    print(f"   Version de Python: {sys.version}")
    print(f"   Version de PyTorch: {torch.__version__}")
    print(f"   Version de NumPy: {np.__version__}")
    
    # Étape 8: Exécuter le script test_widerface.py avec des logs détaillés
    print("\n8. Exécution du script test_widerface.py avec logs détaillés")
    try:
        # Sauvegarder le répertoire courant
        current_dir = os.getcwd()
        
        # Créer un répertoire temporaire pour les résultats de test
        test_results_dir = os.path.join(yolo_dir, "debug_results")
        os.makedirs(test_results_dir, exist_ok=True)
        
        # Créer une copie du script test_widerface.py avec des logs supplémentaires
        debug_script_path = os.path.join(yolo_dir, "test_widerface_debug.py")
        shutil.copy(test_widerface_path, debug_script_path)
        
        # Ajouter des logs supplémentaires
        with open(debug_script_path, 'r') as f:
            script_content = f.read()
        
        # Insérer des affichages de débogage
        debug_imports = "import traceback\nfrom tqdm import tqdm"
        if debug_imports not in script_content:
            script_content = script_content.replace("import os", f"import os\n{debug_imports}")
        
        # Ajouter des logs dans la fonction detect
        detect_func = "def detect(model, img0):"
        detect_func_debug = "def detect(model, img0):\n    print(f\"Détection sur image de forme {img0.shape}\")"
        script_content = script_content.replace(detect_func, detect_func_debug)
        
        # Ajouter des logs dans la boucle principale
        main_loop = "for image_path in tqdm(glob.glob(os.path.join(testset_folder, '*'))):"
        main_loop_debug = """for image_path in tqdm(glob.glob(os.path.join(testset_folder, '*'))):
            try:
                print(f\"Traitement de {image_path}\")\n"""
        script_content = script_content.replace(main_loop, main_loop_debug)
        
        # Ajouter un bloc try/except à la fin de la boucle
        end_loop = "print('done.')"
        end_loop_debug = """            except Exception as e:
                print(f\"ERREUR lors du traitement de {image_path}: {e}\")
                print(traceback.format_exc())
        print('done.')"""
        script_content = script_content.replace(end_loop, end_loop_debug)
        
        # Écrire le script modifié
        with open(debug_script_path, 'w') as f:
            f.write(script_content)
        
        # Exécuter le script
        os.chdir(yolo_dir)
        
        # Récupérer le chemin du fichier wider_val.txt
        wider_val_path = os.path.join(data_dir, "val", "wider_val.txt")
        if not os.path.exists(wider_val_path):
            print(f"✗ ERREUR: Le fichier wider_val.txt n'existe pas: {wider_val_path}")
            wider_val_path = os.path.join(data_dir, "wider_val.txt")
            if not os.path.exists(wider_val_path):
                print(f"✗ ERREUR: Le fichier wider_val.txt n'existe pas: {wider_val_path}")
                # Chercher le fichier
                print("   Recherche du fichier wider_val.txt...")
                found_files = []
                for root, _, files in os.walk(data_dir):
                    for file in files:
                        if file == "wider_val.txt":
                            found_files.append(os.path.join(root, file))
                
                if found_files:
                    print(f"   Fichiers trouvés: {found_files}")
                    wider_val_path = found_files[0]
                else:
                    # Créer un fichier temporaire
                    print("   Création d'un fichier wider_val.txt temporaire...")
                    wider_val_path = os.path.join(test_results_dir, "wider_val.txt")
                    with open(wider_val_path, 'w') as f:
                        # Parcourir les répertoires d'images et créer des entrées
                        for root, _, files in os.walk(val_images_dir):
                            for file in files:
                                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                                    rel_path = os.path.relpath(root, val_images_dir)
                                    if rel_path == '.':
                                        f.write(f"{file}\n")
                                    else:
                                        f.write(f"{rel_path}/{file}\n")
        
        # Exécuter le script
        test_cmd = [
            'python', debug_script_path,
            '--weights', weights_path,
            '--img-size', '640',
            '--conf-thres', '0.02',
            '--dataset_folder', val_images_dir,
            '--save_folder', test_results_dir,
            '--folder_pict', wider_val_path
        ]
        
        print(f"   Commande: {' '.join(test_cmd)}")
        test_output = subprocess.run(test_cmd, capture_output=True, text=True)
        print(f"   Sortie standard: {test_output.stdout}")
        print(f"   Sortie d'erreur: {test_output.stderr}")
        
        # Vérifier les résultats
        if os.path.exists(test_results_dir) and any(f.endswith('.txt') for f in os.listdir(test_results_dir)):
            print(f"✓ test_widerface.py a généré des résultats dans {test_results_dir}")
        else:
            print(f"✗ ERREUR: test_widerface.py n'a pas généré de résultats dans {test_results_dir}")
        
        # Revenir au répertoire d'origine
        os.chdir(current_dir)
    except Exception as e:
        print(f"✗ ERREUR lors de l'exécution de test_widerface.py: {e}")
        print(traceback.format_exc())
    
    # Étape 9: Exécuter le script d'évaluation avec des logs détaillés
    print("\n9. Exécution du script d'évaluation avec logs détaillés")
    try:
        # Aller dans le répertoire d'évaluation
        os.chdir(eval_dir)
        
        # Copier le script d'évaluation avec des logs supplémentaires
        eval_script_path = os.path.join(eval_dir, "evaluation.py")
        debug_eval_script_path = os.path.join(eval_dir, "evaluation_debug.py")
        shutil.copy(eval_script_path, debug_eval_script_path)
        
        # Ajouter des logs supplémentaires
        with open(debug_eval_script_path, 'r') as f:
            eval_content = f.read()
        
        # Ajouter des logs dans la fonction get_preds
        get_preds_func = "def get_preds(pred_dir):"
        get_preds_debug = """def get_preds(pred_dir):
    print(f"Lecture des prédictions depuis {pred_dir}")
    if not os.path.exists(pred_dir):
        print(f"ERREUR: Le répertoire des prédictions n'existe pas: {pred_dir}")
        return {}
    
    events = os.listdir(pred_dir)
    print(f"Nombre d'événements: {len(events)}")"""
        eval_content = eval_content.replace(get_preds_func, get_preds_debug)
        
        # Ajouter des logs dans la fonction read_pred_file
        read_pred_func = "def read_pred_file(filepath):"
        read_pred_debug = """def read_pred_file(filepath):
    print(f"Lecture du fichier de prédiction: {filepath}")
    try:"""
        eval_content = eval_content.replace(read_pred_func, read_pred_debug)
        
        # Ajouter un bloc try/except dans read_pred_file
        read_pred_end = "    return img_file.split('/')[-1], boxes"
        read_pred_debug_end = """        return img_file.split('/')[-1], boxes
    except Exception as e:
        print(f"ERREUR lors de la lecture du fichier {filepath}: {e}")
        print(traceback.format_exc())
        return "", np.array([])"""
        eval_content = eval_content.replace(read_pred_end, read_pred_debug_end)
        
        # Ajouter des logs dans la fonction evaluation
        eval_func = "def evaluation(pred, gt_path, iou_thresh=0.5):"
        eval_debug = """def evaluation(pred, gt_path, iou_thresh=0.5):
    print(f"Évaluation avec pred={pred}, gt_path={gt_path}, iou_thresh={iou_thresh}")
    try:"""
        eval_content = eval_content.replace(eval_func, eval_debug)
        
        # Ajouter un bloc try/except à la fin de la fonction evaluation
        eval_end = "    print(\"==================================================\")"
        eval_debug_end = """    print("==================================================")
    except Exception as e:
        print(f"ERREUR lors de l'évaluation: {e}")
        print(traceback.format_exc())"""
        eval_content = eval_content.replace(eval_end, eval_debug_end)
        
        # Écrire le script modifié
        with open(debug_eval_script_path, 'w') as f:
            f.write(eval_content)
        
        # Exécuter le script
        eval_cmd = [
            'python', debug_eval_script_path,
            '--pred', test_results_dir,
            '--gt', gt_dir
        ]
        
        print(f"   Commande: {' '.join(eval_cmd)}")
        eval_output = subprocess.run(eval_cmd, capture_output=True, text=True)
        print(f"   Sortie standard: {eval_output.stdout}")
        print(f"   Sortie d'erreur: {eval_output.stderr}")
    except Exception as e:
        print(f"✗ ERREUR lors de l'exécution de l'évaluation: {e}")
        print(traceback.format_exc())
    
    # Résumé et recommandations
    print("\n=== RÉSUMÉ ET RECOMMANDATIONS ===")
    print("1. Vérifier que les fichiers .mat de vérité terrain sont correctement chargés")
    print("2. Compiler l'extension bbox avec la bonne version de Python")
    print("3. S'assurer que les fichiers de prédiction sont au bon format")
    print("4. Vérifier la structure des répertoires d'images et de résultats")

# Exemple d'utilisation
if __name__ == "__main__":
    # Configuration
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
    
    # Exécuter le débogage
    debug_evaluation(yolo_dir, weights_path, data_dir, results_dir)
