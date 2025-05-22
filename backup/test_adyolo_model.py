#!/usr/bin/env python3
"""
Test script pour valider l'architecture ADYOLOv5-Face
"""

import torch
import sys
import os

# Ajouter le répertoire courant au path pour les imports
sys.path.append('.')

print("Testing ADYOLOv5-Face architecture...")

try:
    print("1. Testing imports...")
    from models.yolo import Model
    print("   ✓ Model imported successfully")
    
    print("2. Loading ADYOLOv5-Face configuration...")
    model_path = 'models/adyolov5s_simple.yaml'
    
    if os.path.exists(model_path):
        print(f"   ✓ Configuration file found: {model_path}")
        
        # Créer le modèle
        print("3. Creating ADYOLOv5-Face model...")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"   Using device: {device}")
        
        model = Model(model_path, ch=3, nc=1)  # 3 channels input, 1 class (face)
        model = model.to(device)
        
        print("   ✓ Model created successfully")
        print(f"   Model has {len(model.detect_layers)} detection head(s)")
        
        if len(model.detect_layers) == 4:
            print("   ✓ ADYOLOv5-Face architecture confirmed (4 detection heads)")
        else:
            print(f"   ⚠ Expected 4 detection heads, got {len(model.detect_layers)}")
        
        print("4. Testing forward pass...")
        model.eval()
        
        # Test avec une image de test
        test_input = torch.randn(1, 3, 640, 640).to(device)
        
        with torch.no_grad():
            output = model(test_input)
            
        print("   ✓ Forward pass successful")
        
        if isinstance(output, tuple):
            pred, features = output
            print(f"   Output shape: {pred.shape}")
            print(f"   Features: {len(features) if isinstance(features, list) else 'Single tensor'}")
        else:
            print(f"   Output shape: {output.shape}")
        
        print("5. Testing model info...")
        model.info(verbose=False, img_size=640)
        
        print("\n✅ ADYOLOv5-Face model validation successful!")
        print("🎉 Ready for training!")
        
    else:
        print(f"   ❌ Configuration file not found: {model_path}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    print("\n🔧 Check the error above and fix any issues.")
