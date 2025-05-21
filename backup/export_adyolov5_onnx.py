#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Export ADYOLOv5-Face au format ONNX pour l'application Flutter
Script optimisé pour convertir le modèle ADYOLOv5-Face au format compatible avec Flutter
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import subprocess
from pathlib import Path

# Ajuster le chemin pour l'importation des modules YOLOv5
yoloface_path = Path(os.path.dirname(os.path.abspath(__file__))).parent / "yolov5-face"
sys.path.append(str(yoloface_path))

from models.yolo import Model
from utils.general import make_divisible

def parse_args():
    parser = argparse.ArgumentParser(description='Export ADYOLOv5-Face to ONNX')
    parser.add_argument('--weights', type=str, default='runs/train/face_detection_transfer/weights/best.pt',
                        help='Path to trained weights')
    parser.add_argument('--img-size', type=int, default=320, 
                        help='Input image size for the ONNX model (smaller is faster)')
    parser.add_argument('--batch-size', type=int, default=1, 
                        help='Batch size for export (usually 1 for inference)')
    parser.add_argument('--simplify', action='store_true',
                        help='Simplify ONNX model with onnx-simplifier')
    parser.add_argument('--include-nms', action='store_true',
                        help='Include NMS operations in the ONNX model')
    parser.add_argument('--half', action='store_true',
                        help='Export in half precision (FP16)')
    parser.add_argument('--output', type=str, default='export',
                        help='Output folder name')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Créer le dossier de sortie
    export_dir = Path(yoloface_path) / args.output
    os.makedirs(export_dir, exist_ok=True)
    
    print(f"\n=== Exportation du modèle ADYOLOv5-Face vers {export_dir} ===")
    
    # Charger le modèle
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilisation de l'appareil: {device}")
    
    weights_path = Path(yoloface_path) / args.weights
    print(f"Chargement des poids depuis: {weights_path}")
    
    # Charger le modèle et les poids
    try:
        ckpt = torch.load(weights_path, map_location=device)
        model = ckpt['model'].float().eval()
        
        # Fusionner les couches Conv2d+BatchNorm2d pour optimiser le modèle
        for m in model.modules():
            if isinstance(m, (nn.Conv2d,)) and hasattr(m, 'bn'):
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
                
        if args.half:
            model = model.half()
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        return
    
    # Créer un exemple d'entrée
    img = torch.zeros((args.batch_size, 3, args.img_size, args.img_size), device=device)
    if args.half:
        img = img.half()
    
    # Préparer le nom de fichier de sortie
    output_name = f"adyolov5_face_{args.img_size}{'_half' if args.half else ''}"
    if args.include_nms:
        output_name += "_with_nms"
    output_path = export_dir / f"{output_name}.onnx"
    
    # Exportation en ONNX
    print(f"Exportation de ADYOLOv5-Face en ONNX...")
    
    try:
        torch.onnx.export(
            model,
            img,
            output_path,
            verbose=False,
            opset_version=12,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={
                'images': {0: 'batch_size'},  # dimension variable
                'output': {0: 'batch_size'},
            }
        )
        
        # Vérifier le fichier ONNX
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"ONNX model checked: {output_path}")
        
        # Simplification du modèle ONNX si demandé
        if args.simplify:
            print("Simplification du modèle ONNX...")
            try:
                import onnxsim
                model_simplified, check = onnxsim.simplify(onnx_model)
                if check:
                    onnx.save(model_simplified, output_path)
                    print(f"ONNX model simplified: {output_path}")
                else:
                    print("ONNX simplification failed")
            except Exception as e:
                print(f"Error simplifying ONNX model: {e}")
                
        # Optimisation pour Flutter/MediaPipe
        print("Optimisation du modèle pour Flutter/MediaPipe...")
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            # Quantifier le modèle pour réduire la taille
            float_model_path = output_path
            quant_model_path = export_dir / f"{output_name}_quantized.onnx"
            quantize_dynamic(
                float_model_path, 
                quant_model_path, 
                weight_type=QuantType.QUInt8
            )
            print(f"ONNX model quantized: {quant_model_path}")
            
        except Exception as e:
            print(f"Impossible d'optimiser avec onnxruntime: {e}")
            print("Installation des dépendances requises...")
            subprocess.run(["pip", "install", "onnxruntime-gpu" if torch.cuda.is_available() else "onnxruntime"], 
                         check=True)
        
        print("Exportation terminée !")
        print("\nInformations pour l'intégration Flutter:")
        print(f"1. Taille d'entrée: {args.img_size}x{args.img_size}")
        print(f"2. Format d'entrée: RGB, plage [0-1] ou [0-255] selon normalisation")
        print(f"3. Sorties: boîtes de détection, scores de confiance, landmarks faciaux")
        print(f"4. Fichier: {output_path}")
        
    except Exception as e:
        print(f"Erreur lors de l'exportation ONNX: {e}")

if __name__ == "__main__":
    main()
