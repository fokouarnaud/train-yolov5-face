#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script principal pour l'entraînement de YOLOv5-Face sur Google Colab
Ce script coordonne l'ensemble du processus d'entraînement, d'évaluation et d'exportation
"""

import os
import argparse
import time
from pathlib import Path

# Import des modules personnalisés
from data_preparation import DataPreparation
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator
from utils import setup_environment, fix_numpy_issue
from pytorch_fix import fix_pytorch_compatibility

def parse_args():
    """Parse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(description='YOLOv5-Face Training Pipeline')
    
    parser.add_argument('--batch-size', type=int, default=32, 
                        help='Taille du batch pour l\'entraînement')
    parser.add_argument('--epochs', type=int, default=300, 
                        help='Nombre d\'epochs d\'entraînement')
    parser.add_argument('--img-size', type=int, default=640, 
                        help='Taille d\'image pour l\'entraînement')
    parser.add_argument('--model-size', type=str, default='s', choices=['s', 'm', 'l', 'x'],
                        help='Taille du modèle YOLOv5 (s, m, l, x)')
    parser.add_argument('--yolo-version', type=str, default='5.0',
                        help='Version de YOLOv5 (par exemple 5.0)')
    parser.add_argument('--skip-train', action='store_true',
                        help='Ignorer l\'étape d\'entraînement')
    parser.add_argument('--skip-evaluation', action='store_true',
                        help='Ignorer l\'étape d\'évaluation')
    parser.add_argument('--skip-export', action='store_true',
                        help='Ignorer l\'étape d\'exportation')
    
    return parser.parse_args()

def main():
    """Point d'entrée principal du script"""
    args = parse_args()
    
    # Configuration de base
    root_dir = '/content'
    yolo_dir = f'{root_dir}/yolov5-face'
    data_dir = f'{yolo_dir}/data/widerface'
    drive_dataset_path = '/content/drive/MyDrive/dataset'
    
    start_time = time.time()
    
    print("=" * 80)
    print("PIPELINE D'ENTRAÎNEMENT YOLOV5-FACE")
    print("=" * 80)
    
    # Étape 1: Configuration de l'environnement
    setup_environment(yolo_dir)
    
    # Étape 2: Préparation des données
    data_prep = DataPreparation(
        root_dir=root_dir,
        yolo_dir=yolo_dir,
        data_dir=data_dir,
        drive_dataset_path=drive_dataset_path
    )
    data_prep.prepare_dataset()
    
    # Étape 3: Corriger les problèmes connus
    fix_numpy_issue(yolo_dir)
    
    # Étape 3.1: Appliquer le patch pour PyTorch 2.6+
    print("\n=== Application du patch pour PyTorch 2.6+ ===")
    # Utilisation du script de correction PyTorch optimisé
    patch_success = fix_pytorch_compatibility()
    
    if not patch_success:
        print("⚠️ ATTENTION: Le patch pour PyTorch 2.6+ n'a pas pu être appliqué!")
        print("Vous devez corriger manuellement le fichier train.py avant de continuer:")
        print("1. Ouvrez le fichier " + os.path.join(yolo_dir, 'train.py'))
        print("2. Trouvez la ligne: torch.load(weights, map_location=device)")
        print("3. Remplacez-la par: torch.load(weights, map_location=device, weights_only=False)")
        print("4. Enregistrez le fichier et relancez le script")
        
        # Demander à l'utilisateur s'il souhaite continuer malgré l'échec du patch
        user_input = input("\nSouhaitez-vous continuer malgré tout? (oui/non): ")
        if user_input.lower() not in ['oui', 'o', 'yes', 'y']:
            print("\nExécution arrêtée par l'utilisateur.")
            return False
    
    # Étape 4: Entraînement du modèle
    if not args.skip_train:
        trainer = ModelTrainer(
            yolo_dir=yolo_dir,
            data_dir=data_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            img_size=args.img_size,
            model_size=args.model_size,
            yolo_version=args.yolo_version
        )
        train_success = trainer.train()
    else:
        train_success = True
        print("\n=== Étape d'entraînement ignorée ===")
    
    # Étape 5: Évaluation et exportation (si l'entraînement a réussi)
    if train_success:
        evaluator = ModelEvaluator(
            root_dir=root_dir,
            yolo_dir=yolo_dir,
            data_dir=data_dir,
            img_size=args.img_size
        )
        
        if not args.skip_evaluation:
            evaluator.evaluate()
        else:
            print("\n=== Étape d'évaluation ignorée ===")
        
        if not args.skip_export:
            evaluator.export()
        else:
            print("\n=== Étape d'exportation ignorée ===")
    
    # Afficher le temps total d'exécution
    end_time = time.time()
    duration = end_time - start_time
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "=" * 80)
    print(f"PIPELINE TERMINÉ EN {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print("=" * 80)
    
    print("\nRésultats et modèles disponibles dans:")
    print(f"- Modèle PyTorch: {yolo_dir}/runs/train/face_detection_transfer/weights/best.pt")
    print(f"- Modèle ONNX: {yolo_dir}/runs/train/face_detection_transfer/weights/best.onnx")
    print(f"- Métriques et logs: {yolo_dir}/runs/train/face_detection_transfer")

if __name__ == "__main__":
    main()
