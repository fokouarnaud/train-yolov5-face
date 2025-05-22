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

# Importer la configuration centralisée
from config import REPO_URL, DEPENDENCIES, DEFAULT_PATHS, INFO_MESSAGES, DEFAULT_TRAINING, MODEL_CONFIGS

# Import des modules personnalisés
from data_preparation import DataPreparation
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator
from utils import setup_environment, fix_numpy_issue
# Le fix PyTorch n'est plus nécessaire car il est intégré dans le dépôt forké
# from pytorch_fix import fix_pytorch_compatibility

def parse_args():
    """Parse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(description='YOLOv5-Face Training Pipeline')
    
    parser.add_argument('--batch-size', type=int, default=DEFAULT_TRAINING["batch_size"], 
                        help='Taille du batch pour l\'entraînement')
    parser.add_argument('--epochs', type=int, default=DEFAULT_TRAINING["epochs"], 
                        help='Nombre d\'epochs d\'entraînement')
    parser.add_argument('--img-size', type=int, default=DEFAULT_TRAINING["img_size"], 
                        help='Taille d\'image pour l\'entraînement')
    parser.add_argument('--model-size', type=str, default=DEFAULT_TRAINING["model_size"], choices=list(MODEL_CONFIGS.keys()),
                        help='Taille du modèle YOLOv5 (n-0.5, n, s, s6, m, m6, l, l6, x, x6, ad) - "ad" pour ADYOLOv5-Face')
    parser.add_argument('--yolo-version', type=str, default=DEFAULT_TRAINING["yolo_version"],
                        help='Version de YOLOv5 (par exemple 5.0)')
    parser.add_argument('--skip-train', action='store_true',
                        help='Ignorer l\'étape d\'entraînement')
    parser.add_argument('--skip-evaluation', action='store_true',
                        help='Ignorer l\'étape d\'évaluation')
    parser.add_argument('--skip-export', action='store_true',
                        help='Ignorer l\'étape d\'exportation')
    parser.add_argument('--memory-optimized', action='store_true',
                        help='Utiliser l\'entraînement optimisé pour la mémoire (batch size réduit, résolution adaptative)')
    
    return parser.parse_args()

def main():
    """Point d'entrée principal du script"""
    args = parse_args()
    
    # Configuration de base
    root_dir = DEFAULT_PATHS["root_dir"]
    yolo_dir = DEFAULT_PATHS["yolo_dir"]
    data_dir = DEFAULT_PATHS["data_dir"]
    drive_dataset_path = DEFAULT_PATHS["drive_dataset_path"]
    
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
    
    # Le patch PyTorch 2.6+ est déjà intégré dans le dépôt forké
    print("\n=== Vérification de la compatibilité PyTorch ===")
    print(INFO_MESSAGES["pytorch_fix"])
    
    # Pour ADYOLOv5-Face - utiliser le fichier unifié
    if args.model_size == 'ad':
        model_yaml = "adyolov5s.yaml"  # Fichier principal unifié avec GDFusion
        print(f"\n=== Configuration ADYOLOv5-Face ===")
        print(f"Architecture: ADYOLOv5 avec mécanisme Gather-and-Distribute")
        print(f"Fichier YAML: {model_yaml}")
        print(f"Têtes de détection: 4 niveaux (P2/P3/P4/P5)")
        
        # Modifier dynamiquement la configuration
        MODEL_CONFIGS['ad']['yaml'] = model_yaml
    
    # Étape 4: Entraînement du modèle
    if not args.skip_train:
        if args.memory_optimized and args.model_size == 'ad':
            # Utiliser l'entraînement optimisé pour ADYOLOv5-Face
            print("\n=== Mode entraînement optimisé mémoire ===\n")
            from train_adyolo_optimized import train_adyolov5_optimized
            try:
                train_adyolov5_optimized()
                train_success = True
            except Exception as e:
                print(f"❌ Erreur entraînement optimisé: {e}")
                train_success = False
        else:
            # Entraînement standard
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
