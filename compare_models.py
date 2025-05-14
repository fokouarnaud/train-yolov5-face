#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script pour comparer différentes architectures YOLOv5-Face
Ce script exécute l'entraînement et l'évaluation sur différentes variantes
de modèles pour permettre la comparaison des performances
"""

import os
import argparse
import subprocess
import time
import csv
from datetime import datetime
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Importer la configuration centralisée
from config import DEFAULT_TRAINING, DEFAULT_PATHS
from model_training import ModelTrainer
from model_evaluation import ModelEvaluator

def parse_args():
    """Parse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(description='Comparaison des architectures YOLOv5-Face')
    
    parser.add_argument('--models', type=str, nargs='+', 
                        default=['n-0.5', 'n', 's', 's6', 'm', 'l'],
                        help='Liste des modèles à comparer')
    parser.add_argument('--epochs', type=int, default=DEFAULT_TRAINING["epochs"],
                        help='Nombre d\'epochs d\'entraînement')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_TRAINING["batch_size"],
                        help='Taille du batch pour l\'entraînement')
    parser.add_argument('--img-size', type=int, default=DEFAULT_TRAINING["img_size"],
                        help='Taille d\'image pour l\'entraînement')
    parser.add_argument('--skip-train', action='store_true',
                        help='Ignorer l\'étape d\'entraînement (évaluer seulement)')
    parser.add_argument('--output-dir', type=str, default='/content/drive/MyDrive/YOLOv5_Face_Results/comparison',
                        help='Répertoire de sortie pour les résultats')
    
    return parser.parse_args()

def setup_output_directory(output_dir):
    """Configure le répertoire de sortie pour les résultats"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_dir = os.path.join(output_dir, f'comparison_{timestamp}')
    os.makedirs(results_dir)
    
    return results_dir

def train_and_evaluate_model(model_size, epochs, batch_size, img_size, yolo_dir, data_dir):
    """Entraîne et évalue un modèle spécifique"""
    start_time = time.time()
    
    results = {
        'model_size': model_size,
        'epochs': epochs,
        'batch_size': batch_size,
        'img_size': img_size,
        'training_success': False,
        'evaluation_success': False,
        'ap_easy': 0.0,
        'ap_medium': 0.0,
        'ap_hard': 0.0,
        'training_time': 0.0,
        'inference_time': 0.0,
        'model_size_mb': 0.0,
        'parameters': 0,
        'error': None
    }
    
    try:
        # Entraînement du modèle
        trainer = ModelTrainer(
            yolo_dir=yolo_dir,
            data_dir=data_dir,
            batch_size=batch_size,
            epochs=epochs,
            img_size=img_size,
            model_size=model_size
        )
        
        train_success = trainer.train()
        results['training_success'] = train_success
        
        if train_success:
            # Évaluation du modèle
            evaluator = ModelEvaluator(
                root_dir=DEFAULT_PATHS["root_dir"],
                yolo_dir=yolo_dir,
                data_dir=data_dir,
                img_size=img_size
            )
            
            eval_results = evaluator.evaluate(return_metrics=True)
            results['evaluation_success'] = eval_results['success']
            
            if eval_results['success']:
                results['ap_easy'] = eval_results.get('AP_easy', 0.0)
                results['ap_medium'] = eval_results.get('AP_medium', 0.0)
                results['ap_hard'] = eval_results.get('AP_hard', 0.0)
                results['inference_time'] = eval_results.get('inference_time', 0.0)
            
            # Obtenir la taille et les paramètres du modèle
            model_path = f'{yolo_dir}/runs/train/face_detection_transfer/weights/best.pt'
            if os.path.exists(model_path):
                results['model_size_mb'] = os.path.getsize(model_path) / (1024 * 1024)
                
                # Exécuter un script Python pour obtenir le nombre de paramètres
                try:
                    cmd = [
                        'python', '-c', 
                        f"import torch; model = torch.load('{model_path}', map_location='cpu', weights_only=False); "
                        f"print(sum(p.numel() for p in model['model'].parameters()))"
                    ]
                    
                    output = subprocess.check_output(cmd, universal_newlines=True).strip()
                    results['parameters'] = int(output)
                except:
                    pass
    
    except Exception as e:
        results['error'] = str(e)
    
    # Calculer le temps total
    end_time = time.time()
    results['training_time'] = end_time - start_time
    
    return results

def save_results(results, results_dir):
    """Sauvegarde les résultats dans un fichier CSV et JSON"""
    # Sauvegarder en CSV
    csv_path = os.path.join(results_dir, 'results.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = results[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    # Sauvegarder en JSON
    json_path = os.path.join(results_dir, 'results.json')
    with open(json_path, 'w') as jsonfile:
        json.dump(results, jsonfile, indent=4)
    
    print(f"✓ Résultats sauvegardés dans {csv_path} et {json_path}")
    
    return csv_path

def generate_comparison_charts(csv_path, results_dir):
    """Génère des graphiques de comparaison"""
    try:
        # Charger les données
        df = pd.read_csv(csv_path)
        
        # Trier par modèle_size selon un ordre personnalisé
        size_order = {'n-0.5': 1, 'n': 2, 's': 3, 's6': 4, 'm': 5, 'm6': 6, 'l': 7, 'l6': 8, 'x': 9, 'x6': 10}
        df['size_order'] = df['model_size'].map(size_order)
        df = df.sort_values('size_order')
        
        # Définir les styles de graphique
        plt.style.use('ggplot')
        
        # Graphique 1: Précision (AP) par modèle
        fig, ax = plt.subplots(figsize=(12, 8))
        bar_width = 0.25
        index = np.arange(len(df))
        
        ax.bar(index - bar_width, df['ap_easy'], bar_width, label='Easy', color='green', alpha=0.7)
        ax.bar(index, df['ap_medium'], bar_width, label='Medium', color='blue', alpha=0.7)
        ax.bar(index + bar_width, df['ap_hard'], bar_width, label='Hard', color='red', alpha=0.7)
        
        ax.set_xlabel('Modèle', fontsize=12)
        ax.set_ylabel('AP (%)', fontsize=12)
        ax.set_title('Précision moyenne (AP) par modèle sur WiderFace', fontsize=14)
        ax.set_xticks(index)
        ax.set_xticklabels(df['model_size'])
        ax.legend()
        
        plt.savefig(os.path.join(results_dir, 'precision_comparison.png'), dpi=300, bbox_inches='tight')
        
        # Graphique 2: Temps d'inférence vs Précision
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Créer une métrique composite (moyenne des 3 APs)
        df['mean_ap'] = (df['ap_easy'] + df['ap_medium'] + df['ap_hard']) / 3
        
        scatter = ax.scatter(df['inference_time'], df['mean_ap'], 
                            s=df['model_size_mb']*5, # Taille proportionnelle à la taille du modèle
                            c=df['size_order'], # Couleur selon l'ordre de taille
                            alpha=0.7, cmap='viridis')
        
        # Ajouter des étiquettes pour chaque point
        for i, txt in enumerate(df['model_size']):
            ax.annotate(txt, (df['inference_time'].iloc[i], df['mean_ap'].iloc[i]), 
                      xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Temps d\'inférence (ms)', fontsize=12)
        ax.set_ylabel('Précision moyenne (%)', fontsize=12)
        ax.set_title('Compromis Précision vs Vitesse', fontsize=14)
        
        # Ajouter une légende pour la taille des points
        handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6, 
                                               num=4, func=lambda s: s/5)
        size_legend = ax.legend(handles, labels, loc="upper right", title="Taille (MB)")
        
        plt.savefig(os.path.join(results_dir, 'speed_vs_accuracy.png'), dpi=300, bbox_inches='tight')
        
        # Graphique 3: Paramètres vs Précision
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Normaliser les paramètres en millions pour une meilleure lisibilité
        df['params_millions'] = df['parameters'] / 1_000_000
        
        ax.scatter(df['params_millions'], df['mean_ap'], 
                 s=100, # Taille fixe
                 c=df['inference_time'], # Couleur selon le temps d'inférence
                 alpha=0.7, cmap='cool')
        
        # Ajouter des étiquettes pour chaque point
        for i, txt in enumerate(df['model_size']):
            ax.annotate(txt, (df['params_millions'].iloc[i], df['mean_ap'].iloc[i]), 
                      xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Nombre de paramètres (millions)', fontsize=12)
        ax.set_ylabel('Précision moyenne (%)', fontsize=12)
        ax.set_title('Précision vs Complexité du modèle', fontsize=14)
        
        plt.colorbar(ax.collections[0], label='Temps d\'inférence (ms)')
        
        plt.savefig(os.path.join(results_dir, 'params_vs_accuracy.png'), dpi=300, bbox_inches='tight')
        
        print(f"✓ Graphiques de comparaison générés dans {results_dir}")
        return True
    
    except Exception as e:
        print(f"✗ Erreur lors de la génération des graphiques: {e}")
        return False

def print_comparison_table(results):
    """Affiche un tableau comparatif des résultats"""
    print("\n=== COMPARAISON DES MODÈLES YOLOV5-FACE ===\n")
    
    # En-tête
    headers = ["Modèle", "AP Easy", "AP Medium", "AP Hard", "Inférence (ms)", "Taille (MB)", "Params (M)"]
    
    # Calculer les largeurs des colonnes
    col_widths = [max(len(h), max([len(str(r[h.lower().replace(' ', '_').replace('(', '').replace(')', '')])) 
                                 if h.lower().replace(' ', '_').replace('(', '').replace(')', '') in r 
                                 else 0 for r in results] + [0])) 
                 for h in headers]
    
    # Ajuster pour les unités dans les en-têtes
    for i, h in enumerate(headers):
        if '(' in h:
            col_widths[i] = max(col_widths[i], len(h))
    
    # Imprimer l'en-tête
    header_row = " | ".join(f"{h:{w}s}" for h, w in zip(headers, col_widths))
    print(header_row)
    print("-" * len(header_row))
    
    # Trier les résultats par modèle_size selon un ordre personnalisé
    size_order = {'n-0.5': 1, 'n': 2, 's': 3, 's6': 4, 'm': 5, 'm6': 6, 'l': 7, 'l6': 8, 'x': 9, 'x6': 10}
    sorted_results = sorted(results, key=lambda r: size_order.get(r['model_size'], 99))
    
    # Imprimer les résultats
    for r in sorted_results:
        if not r['evaluation_success']:
            continue  # Ignorer les modèles sans évaluation réussie
            
        # Formater les valeurs
        model = r['model_size']
        ap_easy = f"{r['ap_easy']:.2f}%"
        ap_medium = f"{r['ap_medium']:.2f}%"
        ap_hard = f"{r['ap_hard']:.2f}%"
        inference = f"{r['inference_time']:.2f}"
        size_mb = f"{r['model_size_mb']:.2f}"
        params_m = f"{r['parameters']/1_000_000:.2f}"
        
        # Imprimer la ligne
        row = [model, ap_easy, ap_medium, ap_hard, inference, size_mb, params_m]
        print(" | ".join(f"{str(val):{w}s}" for val, w in zip(row, col_widths)))
    
    print("\n")

def main():
    """Point d'entrée principal du script"""
    args = parse_args()
    
    # Configurer le répertoire de sortie
    results_dir = setup_output_directory(args.output_dir)
    print(f"Les résultats seront sauvegardés dans: {results_dir}")
    
    # Paramètres de base
    yolo_dir = DEFAULT_PATHS["yolo_dir"]
    data_dir = DEFAULT_PATHS["data_dir"]
    
    # Liste pour stocker les résultats
    all_results = []
    
    print("\n=== COMPARAISON DES ARCHITECTURES YOLOV5-FACE ===")
    print(f"Modèles à comparer: {', '.join(args.models)}")
    print(f"Paramètres: {args.epochs} epochs, batch size {args.batch_size}, image size {args.img_size}\n")
    
    # Entraîner et évaluer chaque modèle
    for model_size in args.models:
        print(f"\n{'='*20} Modèle: {model_size} {'='*20}")
        
        if not args.skip_train:
            print(f"Entraînement et évaluation de YOLOv5{model_size}...")
            results = train_and_evaluate_model(
                model_size=model_size,
                epochs=args.epochs,
                batch_size=args.batch_size,
                img_size=args.img_size,
                yolo_dir=yolo_dir,
                data_dir=data_dir
            )
        else:
            print(f"Évaluation seulement (entraînement ignoré) pour YOLOv5{model_size}...")
            # TODO: Implémenter l'évaluation sans entraînement
            results = {
                'model_size': model_size,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'img_size': args.img_size,
                'training_success': True,  # On prétend que l'entraînement a réussi
                'evaluation_success': False,
                'ap_easy': 0.0,
                'ap_medium': 0.0,
                'ap_hard': 0.0,
                'training_time': 0.0,
                'inference_time': 0.0,
                'model_size_mb': 0.0,
                'parameters': 0,
                'error': "Entraînement ignoré"
            }
        
        all_results.append(results)
        
        if results['error']:
            print(f"✗ Erreur lors du traitement de YOLOv5{model_size}: {results['error']}")
        else:
            print(f"✓ Traitement de YOLOv5{model_size} terminé")
            if results['evaluation_success']:
                print(f"  AP: {results['ap_easy']:.2f}% (Easy), {results['ap_medium']:.2f}% (Medium), {results['ap_hard']:.2f}% (Hard)")
                print(f"  Temps d'inférence: {results['inference_time']:.2f} ms")
                print(f"  Taille du modèle: {results['model_size_mb']:.2f} MB")
                print(f"  Nombre de paramètres: {results['parameters']:,}")
    
    # Sauvegarder les résultats
    csv_path = save_results(all_results, results_dir)
    
    # Générer les graphiques de comparaison
    generate_comparison_charts(csv_path, results_dir)
    
    # Afficher la table de comparaison
    print_comparison_table(all_results)
    
    print(f"\n✓ Comparaison terminée! Résultats sauvegardés dans {results_dir}")

if __name__ == "__main__":
    main()
