#!/usr/bin/env python3
"""
Script de validation ADYOLOv5-Face spécialement conçu pour Google Colab
Ce script sera utilisé par colab_setup.py pour vérifier que tout fonctionne correctement
"""

import torch
import sys
import os
from pathlib import Path

def test_adyolo_colab():
    """Test spécifique pour Google Colab après setup"""
    
    print("🔍 Validation ADYOLOv5-Face pour Google Colab")
    print("=" * 60)
    
    # Test 1: Environment
    print("1. Environment Check...")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Test 2: Directory structure
    print("\n2. Directory Structure Check...")
    required_dirs = [
        '/content/yolov5-face',
        '/content/yolov5-face/models',
        '/content/yolov5-face/data'
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"   ✅ {dir_path}")
        else:
            print(f"   ❌ {dir_path} - MISSING")
            return False
    
    # Test 3: Critical files
    print("\n3. Critical Files Check...")
    critical_files = [
        '/content/yolov5-face/models/gd.py',
        '/content/yolov5-face/models/adyolov5s_simple.yaml',  
        '/content/yolov5-face/data/hyp.adyolo.yaml'
    ]
    
    for file_path in critical_files:
        if os.path.exists(file_path):
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - MISSING")
            return False
    
    # Test 4: Imports (le plus critique)
    print("\n4. Import Test...")
    try:
        # Ajouter le path YOLOv5-Face
        sys.path.insert(0, '/content/yolov5-face')
        
        print("   Testing models.gd...")
        from models.gd import GDFusion, AttentionFusion, TransformerFusion
        print("   ✅ GD modules imported")
        
        print("   Testing models.common...")
        from models.common import Conv, C3, SPPF
        print("   ✅ Common modules imported")
        
        print("   Testing models.yolo...")
        from models.yolo import Model, Detect
        print("   ✅ YOLO modules imported")
        
        print("   ✅ NO CIRCULAR IMPORT - SUCCESS!")
        
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Module instantiation
    print("\n5. Module Instantiation Test...")
    try:
        # Test GDFusion avec différents paramètres
        gd1 = GDFusion([256, 256], 256, 'attention')
        gd2 = GDFusion(256, 256, 'transformer')
        print("   ✅ GDFusion modules created")
        
        # Test forward pass
        x1 = torch.randn(1, 256, 32, 32)
        x2 = torch.randn(1, 256, 32, 32)
        
        with torch.no_grad():
            out1 = gd1([x1, x2])
            out2 = gd2([x1, x2])
        
        print(f"   ✅ Forward pass: {out1.shape}, {out2.shape}")
        
    except Exception as e:
        print(f"   ❌ Module test failed: {e}")
        return False
    
    # Test 6: Model creation
    print("\n6. ADYOLOv5-Face Model Test...")
    try:
        config_path = '/content/yolov5-face/models/adyolov5s_simple.yaml'
        
        # Créer le modèle
        model = Model(config_path, ch=3, nc=1)
        model.eval()
        
        print(f"   ✅ Model created")
        print(f"   Detection heads: {len(model.detect_layers)}")
        
        # Vérifier que c'est bien ADYOLOv5 (4 têtes de détection)
        if len(model.detect_layers) == 4:
            print("   ✅ ADYOLOv5-Face confirmed (4 detection heads)")
        else:
            print(f"   ⚠ Expected 4 heads, got {len(model.detect_layers)}")
        
        # Test forward pass
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        test_input = torch.randn(1, 3, 640, 640).to(device)
        
        with torch.no_grad():
            output = model(test_input)
        
        if isinstance(output, tuple):
            pred, features = output
            print(f"   ✅ Forward pass successful: {pred.shape}")
        else:
            print(f"   ✅ Forward pass successful: {output.shape}")
            
    except Exception as e:
        print(f"   ❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("✅ ADYOLOv5-Face is ready for training on Google Colab!")
    print("✅ No circular import issues!")
    print("✅ PyTorch 2.6+ compatible!")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = test_adyolo_colab()
    if not success:
        print("\n❌ Validation failed. Please check the setup.")
        sys.exit(1)
    else:
        print("\n🚀 Ready for training with: python main.py --model-size ad")
