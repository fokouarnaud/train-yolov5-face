#!/usr/bin/env python3
"""
Script d'entraînement optimisé pour ADYOLOv5-Face avec gestion mémoire
Résout les problèmes CUDA Out of Memory
"""

import os
import sys
import subprocess
import torch
from config import MODEL_CONFIGS, DEFAULT_PATHS

def optimize_gpu_memory():
    """Optimise les paramètres de mémoire GPU"""
    print("🔧 Optimisation de la mémoire GPU...")
    
    # Variables d'environnement pour optimiser CUDA
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    
    # Nettoyer le cache CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"   ✅ GPU détecté: {torch.cuda.get_device_name()}")
        print(f"   📊 Mémoire libre: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("   ❌ Aucun GPU CUDA détecté")

def get_optimal_batch_size():
    """Détermine le batch size optimal selon la mémoire disponible"""
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
        
        if total_memory > 35:  # A100/V100 40GB
            return 16  # Batch size réduit mais sûr
        elif total_memory > 15:  # T4 16GB
            return 8
        elif total_memory > 10:  # GTX 1080 Ti 11GB
            return 4
        else:  # GPU plus petit
            return 2
    else:
        return 4  # CPU fallback

def train_adyolov5_optimized():
    """Lance l'entraînement ADYOLOv5 avec paramètres optimisés"""
    
    optimize_gpu_memory()
    
    model_config = MODEL_CONFIGS['ad']
    yolo_face_path = DEFAULT_PATHS["yolo_dir"]
    optimal_batch_size = get_optimal_batch_size()
    
    print(f"🚀 Démarrage entraînement ADYOLOv5-Face optimisé")
    print(f"   📦 Batch size optimal: {optimal_batch_size}")
    print(f"   🖼️  Résolution: 512px (réduite pour économiser mémoire)")
    print(f"   📁 Modèle: {model_config['yaml']}")
    
    # Commande d'entraînement optimisée
    train_cmd = [
        "python", f"{yolo_face_path}/train.py",
        "--data", f"{yolo_face_path}/data/widerface.yaml",
        "--cfg", f"{yolo_face_path}/models/{model_config['yaml']}",
        "--weights", f"{yolo_face_path}/weights/{model_config['weights']}",
        "--batch-size", str(optimal_batch_size),  # Batch size optimisé
        "--epochs", "50",  # Moins d'epochs pour test initial
        "--img", "512",  # Résolution réduite (au lieu de 640)
        "--hyp", f"{yolo_face_path}/data/hyp.adyolo.yaml",
        "--project", f"{yolo_face_path}/runs/train",
        "--name", "adyolov5_memory_optimized",
        "--exist-ok",
        "--cache",  # Cache les images pour accélérer
        "--device", "0",  # Force GPU 0
        "--workers", "2",  # Moins de workers
        "--patience", "10"  # Early stopping
    ]
    
    print(f"📋 Commande: {' '.join(train_cmd)}")
    
    try:
        # Lancer l'entraînement
        result = subprocess.run(train_cmd, cwd=yolo_face_path, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Entraînement terminé avec succès!")
            print(f"📁 Résultats sauvés dans: {yolo_face_path}/runs/train/adyolov5_memory_optimized")
        else:
            print("❌ Erreur lors de l'entraînement:")
            print(result.stderr)
            
            # Suggestions si échec
            print("\n💡 Suggestions en cas d'erreur mémoire:")
            print("   1. Réduire encore le batch-size (essayer --batch-size 2)")
            print("   2. Réduire la résolution (essayer --img 416)")
            print("   3. Utiliser un modèle plus petit (yolov5n.pt au lieu de yolov5s.pt)")
            
    except Exception as e:
        print(f"❌ Erreur lors de l'exécution: {e}")

def test_memory_simple():
    """Test simple pour vérifier si les modules GD fonctionnent sans erreur mémoire"""
    print("🧪 Test des modules GD optimisés...")
    
    try:
        # Import et test des modules
        yolo_face_path = DEFAULT_PATHS["yolo_dir"]
        sys.path.append(yolo_face_path)
        from models.gd import GDFusion, AttentionFusion, TransformerFusion
        
        # Test avec tenseurs de petite taille
        test_tensor = torch.randn(1, 128, 32, 32)  # Batch=1, plus petit
        
        if torch.cuda.is_available():
            test_tensor = test_tensor.cuda()
        
        # Test AttentionFusion
        attn_fusion = AttentionFusion(128)
        if torch.cuda.is_available():
            attn_fusion = attn_fusion.cuda()
        
        output_attn = attn_fusion((test_tensor, test_tensor))
        print(f"   ✅ AttentionFusion OK: {output_attn.shape}")
        
        # Test TransformerFusion  
        trans_fusion = TransformerFusion(128)
        if torch.cuda.is_available():
            trans_fusion = trans_fusion.cuda()
            
        output_trans = trans_fusion((test_tensor, test_tensor))
        print(f"   ✅ TransformerFusion OK: {output_trans.shape}")
        
        # Test GDFusion
        gd_fusion = GDFusion(128, 128, 'attention')
        if torch.cuda.is_available():
            gd_fusion = gd_fusion.cuda()
            
        output_gd = gd_fusion(test_tensor)
        print(f"   ✅ GDFusion OK: {output_gd.shape}")
        
        print("✅ Tous les modules GD fonctionnent correctement!")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🎯 ADYOLOv5-Face - Entraînement Optimisé Mémoire")
    print("=" * 60)
    
    # Test des modules avant entraînement
    if test_memory_simple():
        # Si les modules fonctionnent, lancer l'entraînement
        train_adyolov5_optimized()
    else:
        print("❌ Les modules GD ont des erreurs, vérifiez l'implémentation")
