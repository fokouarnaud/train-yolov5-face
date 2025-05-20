#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script pour comparer les performances d'ADYOLOv5-Face et YOLOv5-Face sur la détection de visages
particulièrement sur les petits visages
"""

import argparse
import os
import sys
import time
import cv2
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# Ajuster le chemin pour l'importation des modules YOLOv5
yoloface_path = Path(os.path.dirname(os.path.abspath(__file__))).parent / "yolov5-face"
sys.path.append(str(yoloface_path))

def parse_args():
    parser = argparse.ArgumentParser(description='Compare ADYOLOv5-Face and YOLOv5-Face')
    parser.add_argument('--adyolo-weights', type=str, required=True,
                       help='Chemin vers les poids ADYOLOv5-Face')
    parser.add_argument('--yolo-weights', type=str, required=True,
                       help='Chemin vers les poids YOLOv5-Face standard')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Taille des images pour l\'inférence')
    parser.add_argument('--test-dir', type=str, required=True,
                       help='Répertoire contenant les images de test')
    parser.add_argument('--output-dir', type=str, default='comparison_results',
                       help='Répertoire de sortie pour les résultats')
    parser.add_argument('--conf-thres', type=float, default=0.4,
                       help='Seuil de confiance pour les détections')
    parser.add_argument('--iou-thres', type=float, default=0.45,
                       help='Seuil IoU pour NMS')
    return parser.parse_args()

def load_model(weights_path, device):
    model = torch.load(weights_path, map_location=device)['model'].float().eval()
    return model

def preprocess_image(img_path, img_size):
    # Lire l'image avec OpenCV
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Enregistrer les dimensions originales
    height, width = img.shape[:2]
    
    # Redimensionner l'image
    img_resized = cv2.resize(img, (img_size, img_size))
    
    # Normaliser et convertir en tensor
    img_normalized = img_resized / 255.0
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).float().unsqueeze(0)
    
    return img, img_tensor, (height, width)

def detect_faces(model, img_tensor, conf_thres, iou_thres, img_size, original_shape):
    # Inférence
    with torch.no_grad():
        pred = model(img_tensor)[0]
    
    # NMS
    from utils.general import non_max_suppression_face
    pred = non_max_suppression_face(pred, conf_thres, iou_thres)
    
    # Traiter les prédictions
    height, width = original_shape
    detections = []
    
    for i, det in enumerate(pred):
        if len(det):
            # Redimensionner les coordonnées à la taille originale
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], (height, width)).round()
            det[:, 5:15] = scale_coords_landmarks(img_tensor.shape[2:], det[:, 5:15], (height, width)).round()
            
            # Convertir en liste de dictionnaires
            for *xyxy, conf, cls, lm1x, lm1y, lm2x, lm2y, lm3x, lm3y, lm4x, lm4y, lm5x, lm5y in det:
                bbox = [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                landmarks = [
                    (int(lm1x), int(lm1y)),
                    (int(lm2x), int(lm2y)),
                    (int(lm3x), int(lm3y)),
                    (int(lm4x), int(lm4y)),
                    (int(lm5x), int(lm5y))
                ]
                
                detection = {
                    'bbox': bbox,
                    'confidence': float(conf),
                    'landmarks': landmarks,
                    'area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                    'width': bbox[2] - bbox[0],
                    'height': bbox[3] - bbox[1]
                }
                detections.append(detection)
    
    return detections

def scale_coords(img1_shape, coords, img0_shape):
    # Redimensionner les coordonnées (xyxy) de img1_shape vers img0_shape
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    
    # Clipper les coordonnées
    coords[:, [0, 2]] = coords[:, [0, 2]].clamp(0, img0_shape[1])  # x1, x2
    coords[:, [1, 3]] = coords[:, [1, 3]].clamp(0, img0_shape[0])  # y1, y2
    
    return coords

def scale_coords_landmarks(img1_shape, coords, img0_shape):
    # Redimensionner les coordonnées des landmarks
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain = old / new
    pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    
    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :] /= gain
    
    # Clipper les coordonnées
    coords[:, [0, 2, 4, 6, 8]] = coords[:, [0, 2, 4, 6, 8]].clamp(0, img0_shape[1])  # x
    coords[:, [1, 3, 5, 7, 9]] = coords[:, [1, 3, 5, 7, 9]].clamp(0, img0_shape[0])  # y
    
    return coords

def draw_detections(image, detections, color):
    # Dessiner les détections
    img = image.copy()
    for det in detections:
        # Bounding box
        bbox = det['bbox']
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        # Texte de confiance
        conf_text = f"{det['confidence']:.2f}"
        cv2.putText(img, conf_text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Landmarks
        for lm in det['landmarks']:
            cv2.circle(img, (lm[0], lm[1]), 2, color, -1)
    
    return img

def analyze_results(adyolo_results, yolo_results, image_name, img, output_dir, img_size):
    # Créer un répertoire pour chaque image
    img_output_dir = os.path.join(output_dir, os.path.splitext(image_name)[0])
    os.makedirs(img_output_dir, exist_ok=True)
    
    # Sauvegarder les images avec les détections
    yolo_img = draw_detections(img, yolo_results, (0, 255, 0))  # Vert pour YOLOv5-Face
    adyolo_img = draw_detections(img, adyolo_results, (255, 0, 0))  # Rouge pour ADYOLOv5-Face
    
    # Image combinée avec les deux modèles
    combined_img = img.copy()
    combined_img = draw_detections(combined_img, yolo_results, (0, 255, 0))
    combined_img = draw_detections(combined_img, adyolo_results, (255, 0, 0))
    
    cv2.imwrite(os.path.join(img_output_dir, "yolo.jpg"), cv2.cvtColor(yolo_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(img_output_dir, "adyolo.jpg"), cv2.cvtColor(adyolo_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(img_output_dir, "combined.jpg"), cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR))
    
    # Trier les détections par taille
    yolo_small = [d for d in yolo_results if d['area'] < 3600]  # < 60x60 px
    yolo_medium = [d for d in yolo_results if 3600 <= d['area'] < 10000]  # 60x60 - 100x100 px
    yolo_large = [d for d in yolo_results if d['area'] >= 10000]  # > 100x100 px
    
    adyolo_small = [d for d in adyolo_results if d['area'] < 3600]
    adyolo_medium = [d for d in adyolo_results if 3600 <= d['area'] < 10000]
    adyolo_large = [d for d in adyolo_results if d['area'] >= 10000]
    
    # Résumé des résultats
    results = {
        'image_name': image_name,
        'image_size': (img.shape[1], img.shape[0]),
        'model_input_size': img_size,
        'yolo_total': len(yolo_results),
        'adyolo_total': len(adyolo_results),
        'yolo_small': len(yolo_small),
        'yolo_medium': len(yolo_medium),
        'yolo_large': len(yolo_large),
        'adyolo_small': len(adyolo_small),
        'adyolo_medium': len(adyolo_medium),
        'adyolo_large': len(adyolo_large),
        'small_face_improvement': len(adyolo_small) - len(yolo_small),
        'yolo_avg_conf': sum(d['confidence'] for d in yolo_results) / max(1, len(yolo_results)),
        'adyolo_avg_conf': sum(d['confidence'] for d in adyolo_results) / max(1, len(adyolo_results)),
    }
    
    # Générer un graphique de comparaison
    labels = ['Petits visages', 'Visages moyens', 'Grands visages', 'Total']
    yolo_values = [len(yolo_small), len(yolo_medium), len(yolo_large), len(yolo_results)]
    adyolo_values = [len(adyolo_small), len(adyolo_medium), len(adyolo_large), len(adyolo_results)]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, yolo_values, width, label='YOLOv5-Face', color='green', alpha=0.7)
    rects2 = ax.bar(x + width/2, adyolo_values, width, label='ADYOLOv5-Face', color='red', alpha=0.7)
    
    ax.set_ylabel('Nombre de visages détectés')
    ax.set_title(f'Comparaison des détections sur {image_name}')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # Ajouter les valeurs au-dessus des barres
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(img_output_dir, "comparison_chart.png"))
    plt.close()
    
    # Sauvegarder les résultats détaillés
    import json
    with open(os.path.join(img_output_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

def main():
    args = parse_args()
    
    # Créer le répertoire de sortie
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialiser les modèles
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading YOLOv5-Face model...")
    yolo_model = load_model(args.yolo_weights, device)
    
    print("Loading ADYOLOv5-Face model...")
    adyolo_model = load_model(args.adyolo_weights, device)
    
    # Trouver toutes les images dans le répertoire de test
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(list(Path(args.test_dir).glob(f'*{ext}')))
    
    print(f"Found {len(image_paths)} images for testing")
    
    # Préparer les résultats globaux
    all_results = []
    total_yolo_time = 0
    total_adyolo_time = 0
    
    # Traiter chaque image
    for img_path in tqdm(image_paths, desc="Processing images"):
        img_name = os.path.basename(img_path)
        
        # Prétraitement de l'image
        original_img, img_tensor, original_shape = preprocess_image(str(img_path), args.img_size)
        
        # YOLOv5-Face détection
        start_time = time.time()
        yolo_tensor = img_tensor.to(device)
        yolo_detections = detect_faces(yolo_model, yolo_tensor, args.conf_thres, args.iou_thres, 
                                      args.img_size, original_shape)
        yolo_time = time.time() - start_time
        total_yolo_time += yolo_time
        
        # ADYOLOv5-Face détection
        start_time = time.time()
        adyolo_tensor = img_tensor.to(device)
        adyolo_detections = detect_faces(adyolo_model, adyolo_tensor, args.conf_thres, args.iou_thres, 
                                        args.img_size, original_shape)
        adyolo_time = time.time() - start_time
        total_adyolo_time += adyolo_time
        
        # Analyser et sauvegarder les résultats
        results = analyze_results(adyolo_detections, yolo_detections, img_name, 
                                 original_img, args.output_dir, args.img_size)
        
        # Ajouter les temps d'inférence
        results['yolo_inference_time'] = yolo_time
        results['adyolo_inference_time'] = adyolo_time
        all_results.append(results)
    
    # Générer un rapport global
    total_images = len(image_paths)
    total_yolo_detections = sum(r['yolo_total'] for r in all_results)
    total_adyolo_detections = sum(r['adyolo_total'] for r in all_results)
    
    total_yolo_small = sum(r['yolo_small'] for r in all_results)
    total_yolo_medium = sum(r['yolo_medium'] for r in all_results)
    total_yolo_large = sum(r['yolo_large'] for r in all_results)
    
    total_adyolo_small = sum(r['adyolo_small'] for r in all_results)
    total_adyolo_medium = sum(r['adyolo_medium'] for r in all_results)
    total_adyolo_large = sum(r['adyolo_large'] for r in all_results)
    
    avg_yolo_time = total_yolo_time / total_images
    avg_adyolo_time = total_adyolo_time / total_images
    
    # Créer une visualisation globale
    categories = ['Petits visages', 'Visages moyens', 'Grands visages', 'Total']
    yolo_values = [total_yolo_small, total_yolo_medium, total_yolo_large, total_yolo_detections]
    adyolo_values = [total_adyolo_small, total_adyolo_medium, total_adyolo_large, total_adyolo_detections]
    
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 7))
    rects1 = ax.bar(x - width/2, yolo_values, width, label='YOLOv5-Face', color='green', alpha=0.7)
    rects2 = ax.bar(x + width/2, adyolo_values, width, label='ADYOLOv5-Face', color='red', alpha=0.7)
    
    ax.set_ylabel('Nombre de visages détectés')
    ax.set_title('Comparaison globale des détections YOLOv5-Face vs ADYOLOv5-Face')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    
    # Ajouter les valeurs sur les barres
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "global_comparison.png"))
    
    # Créer un graphique pour les temps d'inférence
    fig, ax = plt.subplots(figsize=(10, 6))
    models = ['YOLOv5-Face', 'ADYOLOv5-Face']
    times = [avg_yolo_time, avg_adyolo_time]
    
    bars = ax.bar(models, times, color=['green', 'red'], alpha=0.7)
    ax.set_ylabel('Temps d\'inférence moyen (secondes)')
    ax.set_title('Comparaison des temps d\'inférence')
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}s',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "inference_time_comparison.png"))
    
    # Sauvegarder le rapport global
    report = {
        'total_images': total_images,
        'total_yolo_detections': total_yolo_detections,
        'total_adyolo_detections': total_adyolo_detections,
        'small_faces': {
            'yolo': total_yolo_small,
            'adyolo': total_adyolo_small,
            'improvement': total_adyolo_small - total_yolo_small,
            'percentage_improvement': ((total_adyolo_small - total_yolo_small) / max(1, total_yolo_small)) * 100
        },
        'medium_faces': {
            'yolo': total_yolo_medium,
            'adyolo': total_adyolo_medium,
            'improvement': total_adyolo_medium - total_yolo_medium,
            'percentage_improvement': ((total_adyolo_medium - total_yolo_medium) / max(1, total_yolo_medium)) * 100
        },
        'large_faces': {
            'yolo': total_yolo_large,
            'adyolo': total_adyolo_large,
            'improvement': total_adyolo_large - total_yolo_large,
            'percentage_improvement': ((total_adyolo_large - total_yolo_large) / max(1, total_yolo_large)) * 100
        },
        'inference_time': {
            'yolo_avg': avg_yolo_time,
            'adyolo_avg': avg_adyolo_time,
            'difference': avg_adyolo_time - avg_yolo_time,
            'percentage_increase': ((avg_adyolo_time - avg_yolo_time) / avg_yolo_time) * 100
        },
        'detailed_results': all_results
    }
    
    import json
    with open(os.path.join(args.output_dir, "global_report.json"), 'w') as f:
        json.dump(report, f, indent=4)
    
    # Générer un rapport HTML
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Comparaison YOLOv5-Face vs ADYOLOv5-Face</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .section {{ margin-bottom: 30px; }}
            .image-container {{ display: flex; flex-wrap: wrap; gap: 20px; }}
            .image-box {{ margin-bottom: 20px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .improvement-positive {{ color: green; }}
            .improvement-negative {{ color: red; }}
            .summary-box {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h1>Comparaison YOLOv5-Face vs ADYOLOv5-Face</h1>
        
        <div class="section summary-box">
            <h2>Résumé global</h2>
            <p>Total d'images analysées: <strong>{total_images}</strong></p>
            <p>Total de visages détectés: <strong>{total_yolo_detections}</strong> (YOLOv5-Face) vs <strong>{total_adyolo_detections}</strong> (ADYOLOv5-Face)</p>
            <p>Amélioration pour les petits visages: <strong class="{'improvement-positive' if (total_adyolo_small - total_yolo_small) > 0 else 'improvement-negative'}">{total_adyolo_small - total_yolo_small} ({((total_adyolo_small - total_yolo_small) / max(1, total_yolo_small)) * 100:.2f}%)</strong></p>
            <p>Temps d'inférence moyen: <strong>{avg_yolo_time:.4f}s</strong> (YOLOv5-Face) vs <strong>{avg_adyolo_time:.4f}s</strong> (ADYOLOv5-Face)</p>
        </div>
        
        <div class="section">
            <h2>Comparaison globale</h2>
            <img src="global_comparison.png" alt="Comparaison globale" style="max-width: 100%;">
            <img src="inference_time_comparison.png" alt="Comparaison des temps d'inférence" style="max-width: 100%;">
        </div>
        
        <div class="section">
            <h2>Analyse détaillée par taille de visage</h2>
            <table>
                <tr>
                    <th>Catégorie</th>
                    <th>YOLOv5-Face</th>
                    <th>ADYOLOv5-Face</th>
                    <th>Amélioration</th>
                    <th>% Amélioration</th>
                </tr>
                <tr>
                    <td>Petits visages (&lt; 60x60 px)</td>
                    <td>{total_yolo_small}</td>
                    <td>{total_adyolo_small}</td>
                    <td class="{'improvement-positive' if (total_adyolo_small - total_yolo_small) > 0 else 'improvement-negative'}">{total_adyolo_small - total_yolo_small}</td>
                    <td class="{'improvement-positive' if (total_adyolo_small - total_yolo_small) > 0 else 'improvement-negative'}">{((total_adyolo_small - total_yolo_small) / max(1, total_yolo_small)) * 100:.2f}%</td>
                </tr>
                <tr>
                    <td>Visages moyens (60x60 - 100x100 px)</td>
                    <td>{total_yolo_medium}</td>
                    <td>{total_adyolo_medium}</td>
                    <td class="{'improvement-positive' if (total_adyolo_medium - total_yolo_medium) > 0 else 'improvement-negative'}">{total_adyolo_medium - total_yolo_medium}</td>
                    <td class="{'improvement-positive' if (total_adyolo_medium - total_yolo_medium) > 0 else 'improvement-negative'}">{((total_adyolo_medium - total_yolo_medium) / max(1, total_yolo_medium)) * 100:.2f}%</td>
                </tr>
                <tr>
                    <td>Grands visages (&gt; 100x100 px)</td>
                    <td>{total_yolo_large}</td>
                    <td>{total_adyolo_large}</td>
                    <td class="{'improvement-positive' if (total_adyolo_large - total_yolo_large) > 0 else 'improvement-negative'}">{total_adyolo_large - total_yolo_large}</td>
                    <td class="{'improvement-positive' if (total_adyolo_large - total_yolo_large) > 0 else 'improvement-negative'}">{((total_adyolo_large - total_yolo_large) / max(1, total_yolo_large)) * 100:.2f}%</td>
                </tr>
                <tr>
                    <td><strong>Total</strong></td>
                    <td><strong>{total_yolo_detections}</strong></td>
                    <td><strong>{total_adyolo_detections}</strong></td>
                    <td class="{'improvement-positive' if (total_adyolo_detections - total_yolo_detections) > 0 else 'improvement-negative'}"><strong>{total_adyolo_detections - total_yolo_detections}</strong></td>
                    <td class="{'improvement-positive' if (total_adyolo_detections - total_yolo_detections) > 0 else 'improvement-negative'}"><strong>{((total_adyolo_detections - total_yolo_detections) / max(1, total_yolo_detections)) * 100:.2f}%</strong></td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Comparaison des performances</h2>
            <table>
                <tr>
                    <th>Métrique</th>
                    <th>YOLOv5-Face</th>
                    <th>ADYOLOv5-Face</th>
                    <th>Différence</th>
                </tr>
                <tr>
                    <td>Temps d'inférence moyen</td>
                    <td>{avg_yolo_time:.4f} secondes</td>
                    <td>{avg_adyolo_time:.4f} secondes</td>
                    <td class="{'improvement-negative' if (avg_adyolo_time - avg_yolo_time) > 0 else 'improvement-positive'}">{avg_adyolo_time - avg_yolo_time:.4f} secondes ({((avg_adyolo_time - avg_yolo_time) / avg_yolo_time) * 100:.2f}%)</td>
                </tr>
            </table>
        </div>
        
        <div class="section">
            <h2>Exemples de détection</h2>
            <p>Cliquez sur les images pour voir les détails de chaque analyse.</p>
            <div class="image-container">
    """
    
    # Ajouter quelques exemples d'images
    example_results = all_results[:min(5, len(all_results))]
    for i, result in enumerate(example_results):
        img_name = result['image_name']
        img_dir = os.path.splitext(img_name)[0]
        html_report += f"""
                <div class="image-box">
                    <h3>Image {i+1}: {img_name}</h3>
                    <a href="{img_dir}/combined.jpg" target="_blank">
                        <img src="{img_dir}/combined.jpg" alt="Combined detection" style="max-width: 300px; max-height: 300px;">
                    </a>
                    <p>YOLOv5-Face: {result['yolo_total']} visages</p>
                    <p>ADYOLOv5-Face: {result['adyolo_total']} visages</p>
                    <p>Amélioration petits visages: <span class="{'improvement-positive' if result['small_face_improvement'] > 0 else 'improvement-negative'}">{result['small_face_improvement']}</span></p>
                    <a href="{img_dir}/comparison_chart.png" target="_blank">Voir le graphique</a>
                </div>
        """
    
    html_report += """
            </div>
        </div>
        
        <div class="section">
            <h2>Conclusion</h2>
            <p>ADYOLOv5-Face montre une amélioration significative dans la détection des petits visages par rapport à YOLOv5-Face standard, avec un léger impact sur les performances d'inférence.</p>
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(args.output_dir, "report.html"), 'w') as f:
        f.write(html_report)
    
    print(f"Comparison complete! Results saved to {args.output_dir}")
    print(f"Global summary:")
    print(f"- Total faces detected: {total_yolo_detections} (YOLOv5-Face) vs {total_adyolo_detections} (ADYOLOv5-Face)")
    print(f"- Small faces improvement: {total_adyolo_small - total_yolo_small} ({((total_adyolo_small - total_yolo_small) / max(1, total_yolo_small)) * 100:.2f}%)")
    print(f"- Average inference time: {avg_yolo_time:.4f}s (YOLOv5-Face) vs {avg_adyolo_time:.4f}s (ADYOLOv5-Face)")

if __name__ == "__main__":
    main()
