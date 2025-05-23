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
from config import MEMORY_OPTIMIZED_TRAINING
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
    parser.add_argument('--paper-config', action='store_true',
                        help='Utiliser la configuration conforme à l\'article ADYOLOv5-Face (batch=32, epochs=250, img=640)')
    
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
    
    # Configuration des paramètres d'entraînement
    if not args.paper_config and args.model_size == 'ad':
        # Utiliser la configuration optimisée mémoire par défaut pour ADYOLOv5
        print("\n=== Configuration Optimisée Mémoire (par défaut) ===")
        print(f"Batch size: {MEMORY_OPTIMIZED_TRAINING['batch_size']} (au lieu de 32)")
        print(f"Epochs: {MEMORY_OPTIMIZED_TRAINING['epochs']} (au lieu de 250)")
        print(f"Image size: {MEMORY_OPTIMIZED_TRAINING['img_size']} (au lieu de 640)")
        print("Utilise --paper-config pour la configuration conforme à l'article")
        
        # Override des paramètres si pas spécifiés
        if args.batch_size == DEFAULT_TRAINING["batch_size"]:
            args.batch_size = MEMORY_OPTIMIZED_TRAINING["batch_size"]
        if args.epochs == DEFAULT_TRAINING["epochs"]:
            args.epochs = MEMORY_OPTIMIZED_TRAINING["epochs"]
        if args.img_size == DEFAULT_TRAINING["img_size"]:
            args.img_size = MEMORY_OPTIMIZED_TRAINING["img_size"]
    
    elif args.paper_config and args.model_size == 'ad':
        print("\n=== Configuration Conforme à l'Article ADYOLOv5-Face ===")
        print(f"Batch size: {args.batch_size} (conforme à l'article)")
        print(f"Epochs: {args.epochs} (conforme à l'article)")
        print(f"Image size: {args.img_size} (conforme à l'article)")
        print("Hyperparamètres: hyp.adyolo.paper.yaml (SGD, lr=1e-2)")
        
        # Modifier le fichier hyperparamètres pour utiliser celui de l'article
        MODEL_CONFIGS['ad']['hyp'] = 'data/hyp.adyolo.paper.yaml'
    
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
