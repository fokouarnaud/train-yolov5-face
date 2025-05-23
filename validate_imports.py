#!/usr/bin/env python3
"""
Script de validation des corrections d'importation
Test rapide pour s'assurer que toutes les corrections sont bonnes
"""

import sys
import os

def test_imports():
    """Test des imports dans tous les scripts"""
    print("🔍 Test des importations dans les scripts...")
    
    scripts_to_test = [
        'config.py',
        'train_adyolo_optimized.py', 
        'test_gd_quick.py',
        'main.py'
    ]
    
    success = True
    
    for script in scripts_to_test:
        print(f"\n   📄 Test de {script}...")
        try:
            if script == 'config.py':
                from config import MODEL_CONFIGS, DEFAULT_PATHS
                print(f"      ✅ MODEL_CONFIGS disponible avec {len(MODEL_CONFIGS)} modèles")
                print(f"      ✅ DEFAULT_PATHS disponible avec {len(DEFAULT_PATHS)} chemins")
                
            elif script == 'train_adyolo_optimized.py':
                # Test d'import sans exécuter la fonction
                import importlib.util
                spec = importlib.util.spec_from_file_location("train_adyolo_optimized", script)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                print(f"      ✅ {script} se charge sans erreur")
                
            elif script == 'test_gd_quick.py':
                # Test d'import sans exécuter la fonction
                import importlib.util
                spec = importlib.util.spec_from_file_location("test_gd_quick", script)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                print(f"      ✅ {script} se charge sans erreur")
                
            elif script == 'main.py':
                # Test d'import sans exécuter la fonction
                import importlib.util
                spec = importlib.util.spec_from_file_location("main", script)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                print(f"      ✅ {script} se charge sans erreur")
                
        except ImportError as e:
            print(f"      ❌ Erreur d'import dans {script}: {e}")
            success = False
        except Exception as e:
            print(f"      ❌ Autre erreur dans {script}: {e}")
            success = False
    
    return success

def check_config_values():
    """Vérifier que les valeurs de config sont correctes"""
    print("\n🔧 Vérification des valeurs de configuration...")
    
    try:
        from config import MODEL_CONFIGS, DEFAULT_PATHS
        
        # Vérifier ADYOLOv5 config
        ad_config = MODEL_CONFIGS.get('ad')
        if ad_config:
            print(f"   ✅ Configuration ADYOLOv5 trouvée:")
            print(f"      - YAML: {ad_config['yaml']}")
            print(f"      - Weights: {ad_config['weights']}")
            print(f"      - Image size: {ad_config['img_size']}")
        else:
            print("   ❌ Configuration ADYOLOv5 manquante")
            return False
        
        # Vérifier chemins par défaut
        yolo_dir = DEFAULT_PATHS.get("yolo_dir")
        if yolo_dir:
            print(f"   ✅ Chemin YOLOv5-Face: {yolo_dir}")
        else:
            print("   ❌ Chemin YOLOv5-Face manquant")
            return False
            
        return True
        
    except Exception as e:
        print(f"   ❌ Erreur lors de la vérification config: {e}")
        return False

def main():
    """Fonction principale de validation"""
    print("=" * 60)
    print("🔍 VALIDATION CORRECTIONS D'IMPORTATION")
    print("=" * 60)
    
    success = True
    
    # Test 1: Imports
    if not test_imports():
        success = False
    
    # Test 2: Config values
    if not check_config_values():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✅ TOUTES LES CORRECTIONS SONT BONNES!")
        print("🚀 Vous pouvez maintenant utiliser:")
        print("   python main.py --model-size ad --memory-optimized")
        print("   python train_adyolo_optimized.py")
        print("   python test_gd_quick.py")
    else:
        print("❌ CERTAINES CORRECTIONS ONT DES PROBLÈMES!")
        print("🔧 Vérifiez les erreurs ci-dessus")
    print("=" * 60)

if __name__ == "__main__":
    main()
