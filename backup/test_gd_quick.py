#!/usr/bin/env python3
"""
Script de test rapide pour valider les modules GD optimisés
Utiliser ce script pour tester avant l'entraînement complet
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

def test_gd_modules():
    """Test des modules GD optimisés en mémoire"""
    print("🧪 Test des modules GD optimisés pour la mémoire...")
    
    # Ajouter le chemin YOLOv5-Face
    from config import DEFAULT_PATHS
    yolo_path = DEFAULT_PATHS["yolo_dir"]
    if yolo_path not in sys.path:
        sys.path.append(yolo_path)
    
    try:
        # Import des modules
        from models.gd import GDFusion, AttentionFusion, TransformerFusion
        print("   ✅ Import des modules GD réussi")
        
        # Paramètres de test
        batch_size = 2  # Petit batch pour test
        channels = 128
        height, width = 32, 32  # Petite résolution
        
        # Créer tenseur de test
        test_tensor = torch.randn(batch_size, channels, height, width)
        if torch.cuda.is_available():
            test_tensor = test_tensor.cuda()
            print(f"   📱 GPU: {torch.cuda.get_device_name()}")
        
        print(f"   📊 Tensor test: {test_tensor.shape}")
        
        # Test 1: AttentionFusion optimisé
        print("\n   🔍 Test AttentionFusion optimisé...")
        attn_fusion = AttentionFusion(channels)
        if torch.cuda.is_available():
            attn_fusion = attn_fusion.cuda()
        
        # Forward pass
        output_attn = attn_fusion((test_tensor, test_tensor))
        print(f"      ✅ Input: {test_tensor.shape} → Output: {output_attn.shape}")
        
        # Vérifier la mémoire utilisée
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1e6  # MB
            print(f"      📈 Mémoire utilisée: {memory_used:.1f} MB")
        
        # Test 2: TransformerFusion optimisé
        print("\n   🔄 Test TransformerFusion optimisé...")
        trans_fusion = TransformerFusion(channels)
        if torch.cuda.is_available():
            trans_fusion = trans_fusion.cuda()
        
        # Forward pass
        output_trans = trans_fusion((test_tensor, test_tensor))
        print(f"      ✅ Input: {test_tensor.shape} → Output: {output_trans.shape}")
        
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1e6  # MB
            print(f"      📈 Mémoire utilisée: {memory_used:.1f} MB")
        
        # Test 3: GDFusion complet
        print("\n   🎯 Test GDFusion avec attention...")
        gd_fusion_attn = GDFusion(channels, channels, 'attention')
        if torch.cuda.is_available():
            gd_fusion_attn = gd_fusion_attn.cuda()
        
        output_gd_attn = gd_fusion_attn(test_tensor)
        print(f"      ✅ Input: {test_tensor.shape} → Output: {output_gd_attn.shape}")
        
        print("\n   🎯 Test GDFusion avec transformer...")
        gd_fusion_trans = GDFusion(channels, channels, 'transformer')
        if torch.cuda.is_available():
            gd_fusion_trans = gd_fusion_trans.cuda()
        
        output_gd_trans = gd_fusion_trans(test_tensor)
        print(f"      ✅ Input: {test_tensor.shape} → Output: {output_gd_trans.shape}")
        
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1e6  # MB
            print(f"      📈 Mémoire totale utilisée: {memory_used:.1f} MB")
            
            # Nettoyer la mémoire
            torch.cuda.empty_cache()
        
        print("\n✅ Tous les tests des modules GD ont réussi!")
        return True
        
    except Exception as e:
        print(f"\n❌ Erreur lors du test des modules GD: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_loading():
    """Test de chargement du modèle ADYOLOv5"""
    print("\n🏗️ Test de chargement du modèle ADYOLOv5...")
    
    from config import DEFAULT_PATHS
    yolo_path = DEFAULT_PATHS["yolo_dir"]
    sys.path.append(yolo_path)
    
    try:
        from models.yolo import Model
        
        # Tester le chargement du fichier YAML
        yaml_path = f"{yolo_path}/models/adyolov5s.yaml"
        if os.path.exists(yaml_path):
            print(f"   ✅ Fichier YAML trouvé: {yaml_path}")
            
            # Charger le modèle (sans poids pré-entraînés pour test rapide)
            model = Model(yaml_path, ch=3, nc=1)  # 1 classe pour détection de visage
            
            # Test avec un batch plus petit
            test_input = torch.randn(1, 3, 256, 256)  # Résolution réduite
            if torch.cuda.is_available():
                model = model.cuda()
                test_input = test_input.cuda()
            
            print(f"   📊 Input test: {test_input.shape}")
            
            # Forward pass
            with torch.no_grad():
                output = model(test_input)
            
            print(f"   ✅ Modèle chargé et testé avec succès!")
            print(f"   📈 Sorties: {len(output)} têtes de détection")
            for i, out in enumerate(output):
                print(f"      P{i+2}: {out.shape}")
            
            return True
        else:
            print(f"   ❌ Fichier YAML non trouvé: {yaml_path}")
            return False
            
    except Exception as e:
        print(f"   ❌ Erreur lors du test du modèle: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Fonction principale de test"""
    print("=" * 60)
    print("🧪 TEST RAPIDE - Modules GD Optimisés ADYOLOv5")
    print("=" * 60)
    
    # Information système
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"🔥 PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"🚀 CUDA: {torch.version.cuda}")
        print(f"📱 GPU: {torch.cuda.get_device_name()}")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"💾 Mémoire GPU totale: {total_memory:.1f} GB")
    else:
        print("⚠️  CUDA non disponible, utilisation CPU")
    
    # Tests
    success = True
    
    # Test 1: Modules GD
    if not test_gd_modules():
        success = False
    
    # Test 2: Chargement modèle
    if not test_model_loading():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✅ TOUS LES TESTS ONT RÉUSSI!")
        print("🚀 Vous pouvez procéder à l'entraînement ADYOLOv5")
        print("\nCommandes suggérées:")
        print("  python main.py --model-size ad --memory-optimized")
        print("  ou")
        print("  python train_adyolo_optimized.py")
    else:
        print("❌ CERTAINS TESTS ONT ÉCHOUÉ!")
        print("🔧 Vérifiez les erreurs ci-dessus avant l'entraînement")
    print("=" * 60)

if __name__ == "__main__":
    main()
