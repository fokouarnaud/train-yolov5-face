#!/usr/bin/env python3
"""
Test simple des modules GD après correction BatchNorm
"""

import torch
import torch.nn as nn
import sys
import os

# Ajouter le chemin vers models
sys.path.append('../yolov5-face')

def test_gd_modules():
    """Test des modules GD corrigés"""
    print("🧪 Test des modules GD après correction BatchNorm...")
    
    try:
        from models.gd import GDFusion, AttentionFusion, TransformerFusion, ConvNoBN
        
        # Test ConvNoBN avec batch_size=1
        print("\n✅ Test ConvNoBN avec batch_size=1...")
        conv_nobn = ConvNoBN(64, 16, 1)
        test_tensor_1x1 = torch.randn(1, 64, 1, 1)  # Le tensor problématique
        
        if torch.cuda.is_available():
            conv_nobn = conv_nobn.cuda()
            test_tensor_1x1 = test_tensor_1x1.cuda()
        
        output_nobn = conv_nobn(test_tensor_1x1)
        print(f"   ConvNoBN: {test_tensor_1x1.shape} → {output_nobn.shape} ✅")
        
        # Test AttentionFusion avec batch_size=1
        print("\n✅ Test AttentionFusion avec batch_size=1...")
        attn_fusion = AttentionFusion(128)
        test_tensor = torch.randn(1, 128, 32, 32)  # Batch size = 1
        
        if torch.cuda.is_available():
            attn_fusion = attn_fusion.cuda()
            test_tensor = test_tensor.cuda()
        
        output_attn = attn_fusion((test_tensor, test_tensor))
        print(f"   AttentionFusion: {test_tensor.shape} → {output_attn.shape} ✅")
        
        # Test TransformerFusion avec batch_size=1
        print("\n✅ Test TransformerFusion avec batch_size=1...")
        trans_fusion = TransformerFusion(128)
        
        if torch.cuda.is_available():
            trans_fusion = trans_fusion.cuda()
        
        output_trans = trans_fusion((test_tensor, test_tensor))
        print(f"   TransformerFusion: {test_tensor.shape} → {output_trans.shape} ✅")
        
        # Test GDFusion complet avec batch_size=1
        print("\n✅ Test GDFusion complet avec batch_size=1...")
        gd_fusion = GDFusion(128, 128, 'attention')
        
        if torch.cuda.is_available():
            gd_fusion = gd_fusion.cuda()
        
        output_gd = gd_fusion(test_tensor)
        print(f"   GDFusion: {test_tensor.shape} → {output_gd.shape} ✅")
        
        # Test avec batch_size plus grand aussi
        print("\n✅ Test avec batch_size=4 (normal)...")
        test_tensor_batch = torch.randn(4, 128, 32, 32)
        
        if torch.cuda.is_available():
            test_tensor_batch = test_tensor_batch.cuda()
        
        output_batch = gd_fusion(test_tensor_batch)
        print(f"   GDFusion batch=4: {test_tensor_batch.shape} → {output_batch.shape} ✅")
        
        print("\n🎉 TOUS LES TESTS RÉUSSIS!")
        print("Les modules GD fonctionnent maintenant avec batch_size=1")
        return True
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("🔧 TEST CORRECTION BATCHNORM - Modules GD")
    print("=" * 60)
    
    # Information système
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"🔥 PyTorch: {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"🚀 CUDA: {torch.version.cuda}")
        print(f"📱 GPU: {torch.cuda.get_device_name()}")
    else:
        print("⚠️  CUDA non disponible, utilisation CPU")
    
    # Test principal
    success = test_gd_modules()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ CORRECTION BATCHNORM RÉUSSIE!")
        print("🚀 ADYOLOv5-Face peut maintenant s'entraîner sans erreur")
        print("\nCommandes à utiliser:")
        print("  python main.py --model-size ad              # Configuration optimisée")
        print("  python main.py --model-size ad --paper-config  # Configuration article")
    else:
        print("❌ LA CORRECTION A ÉCHOUÉ!")
        print("🔧 Vérifiez les erreurs ci-dessus")
    print("=" * 60)

if __name__ == "__main__":
    main()
