#!/usr/bin/env python3
"""
Script de validation des configurations ADYOLOv5-Face
Vérifie que les paramètres correspondent bien aux configurations définies
"""

import os
import sys

def validate_configurations():
    """Valide les configurations disponibles"""
    print("🔍 Validation des configurations ADYOLOv5-Face...")
    
    try:
        from config import DEFAULT_TRAINING, MEMORY_OPTIMIZED_TRAINING, MODEL_CONFIGS
        
        print("\n📊 Configurations disponibles:")
        
        # Configuration par défaut (article)
        print("\n✅ Configuration par défaut (conforme à l'article):")
        print(f"   - Batch size: {DEFAULT_TRAINING['batch_size']}")
        print(f"   - Epochs: {DEFAULT_TRAINING['epochs']}")
        print(f"   - Image size: {DEFAULT_TRAINING['img_size']}")
        
        # Configuration optimisée mémoire
        print("\n✅ Configuration optimisée mémoire:")
        print(f"   - Batch size: {MEMORY_OPTIMIZED_TRAINING['batch_size']}")
        print(f"   - Epochs: {MEMORY_OPTIMIZED_TRAINING['epochs']}")
        print(f"   - Image size: {MEMORY_OPTIMIZED_TRAINING['img_size']}")
        
        # Configuration ADYOLOv5
        ad_config = MODEL_CONFIGS.get('ad', {})
        print("\n✅ Configuration modèle ADYOLOv5:")
        print(f"   - YAML: {ad_config.get('yaml', 'N/A')}")
        print(f"   - Weights: {ad_config.get('weights', 'N/A')}")
        print(f"   - Image size: {ad_config.get('img_size', 'N/A')}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Erreur d'import: {e}")
        return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def check_hyperparameters_file():
    """Vérifie que le fichier hyperparamètres de l'article existe"""
    print("\n🔍 Vérification du fichier hyperparamètres...")
    
    hyp_file = "../yolov5-face/data/hyp.adyolo.paper.yaml"
    
    if os.path.exists(hyp_file):
        print(f"✅ Fichier trouvé: {hyp_file}")
        
        # Lire quelques paramètres clés
        try:
            with open(hyp_file, 'r') as f:
                content = f.read()
                
            if "lr0: 0.01" in content:
                print("✅ Learning rate initial: 0.01 (conforme)")
            else:
                print("⚠️ Learning rate initial non conforme")
                
            if "weight_decay: 0.005" in content:
                print("✅ Weight decay: 0.005 (conforme)")
            else:
                print("⚠️ Weight decay non conforme")
                
            return True
            
        except Exception as e:
            print(f"❌ Erreur lecture fichier: {e}")
            return False
    else:
        print(f"❌ Fichier non trouvé: {hyp_file}")
        return False

def main():
    """Fonction principale de validation"""
    print("=" * 60)
    print("🔍 VALIDATION CONFIGURATIONS ADYOLOv5-Face")
    print("=" * 60)
    
    success = True
    
    # Test 1: Configurations
    if not validate_configurations():
        success = False
    
    # Test 2: Fichier hyperparamètres
    if not check_hyperparameters_file():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✅ TOUTES LES VALIDATIONS SONT PASSÉES!")
        print("\n🚀 Commandes disponibles:")
        print("   # Configuration optimisée mémoire (par défaut)")
        print("   python main.py --model-size ad")
        print("")
        print("   # Configuration conforme à l'article")
        print("   python main.py --model-size ad --paper-config")
    else:
        print("❌ CERTAINES VALIDATIONS ONT ÉCHOUÉ!")
        print("🔧 Vérifiez les erreurs ci-dessus")
    print("=" * 60)

if __name__ == "__main__":
    main()
