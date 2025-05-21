#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Patch pour assurer la compatibilité avec PyTorch 2.6
Ce script modifie les fichiers nécessaires pour que ADYOLOv5-Face
fonctionne correctement avec PyTorch 2.6+
"""

import os
import re
import sys

def patch_common_py():
    """Patch models/common.py pour la compatibilité PyTorch 2.6+"""
    common_py_path = os.path.join('/content', 'yolov5-face', 'models', 'common.py')
    
    if not os.path.exists(common_py_path):
        print(f"Le fichier {common_py_path} n'existe pas")
        return False
    
    # Lire le fichier
    with open(common_py_path, 'r') as f:
        content = f.read()
    
    # Modifier la classe Conv pour assurer la compatibilité
    # Rechercher le constructeur de la classe Conv
    conv_init_pattern = r'def __init__\(self, c1, c2, k=1, s=1, p=None, g=1, act=True\):'
    conv_init_replacement = 'def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):\n        # Handle c1, c2 being lists/tuples in PyTorch 2.6+'
    
    if conv_init_pattern in content:
        # Ajouter la logique de compatibilité juste après la ligne d'initialisation
        compatibility_code = """
        # Handle c1, c2 being lists/tuples in PyTorch 2.6+
        if isinstance(c1, (list, tuple)):
            if len(c1) >= 1:
                c1 = c1[0]
        if isinstance(c2, (list, tuple)):
            if len(c2) >= 1:
                c2 = c2[0]
        if isinstance(k, (list, tuple)):
            if len(k) >= 1:
                k = k[0]
        if isinstance(s, (list, tuple)):
            if len(s) >= 1:
                s = s[0]
        """
        
        # Trouver la position après la ligne d'initialisation
        pattern = r'def __init__\(self, c1, c2, k=1, s=1, p=None, g=1, act=True\):\s*\n'
        match = re.search(pattern, content)
        if match:
            pos = match.end()
            new_content = content[:pos] + compatibility_code + content[pos:]
            
            # Écrire le contenu modifié
            with open(common_py_path, 'w') as f:
                f.write(new_content)
            
            print(f"✓ Le fichier {common_py_path} a été patché avec succès!")
            return True
        else:
            print(f"❌ Pattern d'initialisation non trouvé dans {common_py_path}")
            return False
    else:
        print(f"❌ Format de classe Conv différent de celui attendu dans {common_py_path}")
        return False

def patch_yolo_py():
    """Patch models/yolo.py pour la compatibilité PyTorch 2.6+"""
    yolo_py_path = os.path.join('/content', 'yolov5-face', 'models', 'yolo.py')
    
    if not os.path.exists(yolo_py_path):
        print(f"Le fichier {yolo_py_path} n'existe pas")
        return False
    
    # Lire le fichier
    with open(yolo_py_path, 'r') as f:
        content = f.read()
    
    # Ajouter un traitement spécial pour les arguments de Conv
    model_parse_pattern = r'def parse_model\(d, ch\):'
    
    if model_parse_pattern in content:
        # Trouver la position juste après la déclaration de la fonction
        pattern = r'def parse_model\(d, ch\):\s*\n'
        match = re.search(pattern, content)
        if match:
            pos = match.end()
            
            # Code de compatibilité à ajouter
            compatibility_code = """
    # PyTorch 2.6+ compatibility
    def normalize_args(args):
        if isinstance(args, list):
            return [normalize_args(a) for a in args]
        elif isinstance(args, tuple):
            return tuple(normalize_args(list(args)))
        elif isinstance(args, dict):
            return {k: normalize_args(v) for k, v in args.items()}
        else:
            return args
            
    """
            
            new_content = content[:pos] + compatibility_code + content[pos:]
            
            # Écrire le contenu modifié
            with open(yolo_py_path, 'w') as f:
                f.write(new_content)
            
            print(f"✓ Le fichier {yolo_py_path} a été patché avec succès!")
            return True
        else:
            print(f"❌ Pattern parse_model non trouvé dans {yolo_py_path}")
            return False
    else:
        print(f"❌ Format de parse_model différent de celui attendu dans {yolo_py_path}")
        return False

if __name__ == "__main__":
    print("Application des correctifs pour la compatibilité PyTorch 2.6+...")
    
    success_common = patch_common_py()
    success_yolo = patch_yolo_py()
    
    if success_common and success_yolo:
        print("✅ Tous les correctifs ont été appliqués avec succès!")
        sys.exit(0)
    else:
        print("❌ Certains correctifs n'ont pas pu être appliqués")
        sys.exit(1)
