#!/usr/bin/env python3
"""
Test script pour vérifier que l'importation circulaire est résolue
"""

print("Testing imports...")

try:
    print("1. Importing models.common...")
    from models.common import Conv, Bottleneck, C3
    print("   ✓ models.common imported successfully")
    
    print("2. Importing models.gd...")
    from models.gd import GDFusion, AttentionFusion, TransformerFusion
    print("   ✓ models.gd imported successfully")
    
    print("3. Importing models.yolo...")
    from models.yolo import Model, Detect
    print("   ✓ models.yolo imported successfully")
    
    print("4. Testing GDFusion instantiation...")
    # Test avec liste de canaux d'entrée
    gd_fusion = GDFusion([256, 256], 256, 'attention')
    print("   ✓ GDFusion created successfully with list input")
    
    # Test avec entier unique
    gd_fusion2 = GDFusion(256, 256, 'transformer')
    print("   ✓ GDFusion created successfully with int input")
    
    print("\n✅ All imports and instantiations successful!")
    print("🎉 Circular import issue resolved!")
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("There's still a circular import issue.")
    
except Exception as e:
    print(f"❌ Other Error: {e}")
    print("There's an issue with module creation.")
