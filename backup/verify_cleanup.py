#!/usr/bin/env python3
"""
Script de vérification du nettoyage ADYOLOv5-Face
Vérifie que toutes les références utilisent maintenant le bon fichier YAML
"""

import os
import re
from pathlib import Path

def check_file_references(directory):
    """Vérifie les références aux fichiers YAML dans tous les scripts"""
    
    print(f"🔍 Vérification des références dans {directory}")
    print("=" * 60)
    
    # Patterns à chercher
    patterns = {
        'adyolov5s_simple': r'adyolov5s_simple\.yaml',
        'adyolov5s': r'adyolov5s\.yaml',
        'old_gatherlayer': r'adyolov5s_old_gatherlayer\.yaml'
    }
    
    results = {'files_checked': 0, 'references_found': {}}
    
    # Parcourir tous les fichiers Python
    for file_path in Path(directory).rglob('*.py'):
        results['files_checked'] += 1
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            for pattern_name, pattern in patterns.items():
                matches = re.findall(pattern, content)
                if matches:
                    if pattern_name not in results['references_found']:
                        results['references_found'][pattern_name] = []
                    results['references_found'][pattern_name].append(str(file_path))
                    
        except Exception as e:
            print(f"⚠️ Erreur lors de la lecture de {file_path}: {e}")
    
    return results

def verify_yaml_files(models_dir):
    """Vérifie la présence et le contenu des fichiers YAML"""
    
    print(f"\n📁 Vérification des fichiers YAML dans {models_dir}")
    print("=" * 60)
    
    # Fichiers attendus
    expected_files = {
        'adyolov5s.yaml': 'Fichier principal ADYOLOv5 avec GDFusion',
        'adyolov5s_old_gatherlayer.yaml': 'Ancien fichier avec GatherLayer (sauvegarde)'
    }
    
    for filename, description in expected_files.items():
        file_path = os.path.join(models_dir, filename)
        
        if os.path.exists(file_path):
            print(f"✅ {filename}: {description}")
            
            # Vérifier le contenu
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                if 'GDFusion' in content:
                    print(f"   🔧 Contient GDFusion")
                elif 'GatherLayer' in content:
                    print(f"   📦 Contient GatherLayer (ancien système)")
                else:
                    print(f"   ❓ Système de fusion non identifié")
                    
            except Exception as e:
                print(f"   ⚠️ Erreur de lecture: {e}")
        else:
            print(f"❌ {filename}: MANQUANT")

def main():
    """Fonction principale de vérification"""
    
    print("🧹 Vérification du Nettoyage ADYOLOv5-Face")
    print("=" * 60)
    
    # Chemins à vérifier
    base_path = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_path, '..', 'yolov5-face', 'models')
    
    # 1. Vérifier les fichiers YAML
    verify_yaml_files(models_dir)
    
    # 2. Vérifier les références dans les scripts
    results = check_file_references(base_path)
    
    print(f"\n📊 Résumé de la vérification")
    print("=" * 60)
    print(f"Fichiers Python vérifiés: {results['files_checked']}")
    
    # Afficher les références trouvées
    if results['references_found']:
        for pattern_name, files in results['references_found'].items():
            print(f"\n🔍 Références '{pattern_name}':")
            for file_path in files:
                print(f"   - {file_path}")
    else:
        print("✅ Aucune référence obsolète trouvée")
    
    # Recommandations
    print(f"\n💡 Recommandations:")
    
    if 'adyolov5s_simple' in results['references_found']:
        print("❌ Des références à 'adyolov5s_simple.yaml' ont été trouvées")
        print("   → Ces références doivent être mises à jour vers 'adyolov5s.yaml'")
    else:
        print("✅ Aucune référence obsolète à 'adyolov5s_simple.yaml'")
    
    if 'adyolov5s' in results['references_found']:
        print("✅ Références correctes à 'adyolov5s.yaml' trouvées")
    else:
        print("⚠️ Aucune référence à 'adyolov5s.yaml' trouvée")
        print("   → Vérifiez que les scripts utilisent bien le bon fichier")
    
    print(f"\n🎯 Structure finale recommandée:")
    print(f"   ✅ adyolov5s.yaml (principal, avec GDFusion)")
    print(f"   📁 adyolov5s_old_gatherlayer.yaml (sauvegarde)")
    print(f"   ❌ adyolov5s_simple.yaml (supprimé)")
    
    print(f"\n🚀 Prêt pour l'entraînement ADYOLOv5-Face !")

if __name__ == "__main__":
    main()
