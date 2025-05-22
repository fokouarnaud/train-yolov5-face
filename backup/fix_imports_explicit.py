#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de correction explicite pour l'erreur d'import dans common.py
"""

import os
import sys

def fix_common_py():
    common_py_path = '/content/yolov5-face/models/common.py'
    
    if os.path.exists(common_py_path):
        print(f"Correction de {common_py_path}...")
        
        with open(common_py_path, 'r') as f:
            content = f.read()
        
        # Vérifier si le fichier commence par une classe sans les imports nécessaires
        if content.startswith('class '):
            # Ajouter les imports au début du fichier
            new_content = 'import torch\nimport torch.nn as nn\nimport warnings\n\n' + content
            
            # Écrire les modifications
            with open(common_py_path, 'w') as f:
                f.write(new_content)
                
            print(f"✓ Imports ajoutés au début de {common_py_path}")
        else:
            # Vérifier si les imports existent déjà mais sont peut-être mal placés
            if 'import torch' in content and 'import torch.nn as nn' in content:
                print(f"✓ Les imports sont déjà présents dans {common_py_path}")
            else:
                # Reconstruire le fichier avec les imports en premier
                lines = content.splitlines()
                import_lines = []
                other_lines = []
                
                for line in lines:
                    if line.startswith('import ') or line.startswith('from '):
                        import_lines.append(line)
                    else:
                        other_lines.append(line)
                
                # Ajouter nos imports spécifiques s'ils ne sont pas déjà présents
                if 'import torch' not in '\n'.join(import_lines):
                    import_lines.insert(0, 'import torch')
                if 'import torch.nn as nn' not in '\n'.join(import_lines):
                    import_lines.insert(1, 'import torch.nn as nn')
                if 'import warnings' not in '\n'.join(import_lines):
                    import_lines.append('import warnings')
                
                # Reconstruire le contenu avec les imports en premier
                new_content = '\n'.join(import_lines) + '\n\n' + '\n'.join(other_lines)
                
                # Écrire les modifications
                with open(common_py_path, 'w') as f:
                    f.write(new_content)
                    
                print(f"✓ Imports réorganisés dans {common_py_path}")
    else:
        print(f"✗ Fichier {common_py_path} introuvable")
        return False
    
    # Vérifier si le problème est résolu
    try:
        sys.path.insert(0, '/content/yolov5-face')
        from models.common import Conv, DWConv
        print("✓ La correction a réussi: les imports fonctionnent maintenant")
        return True
    except Exception as e:
        print(f"✗ La correction n'a pas résolu le problème: {e}")
        return False

if __name__ == "__main__":
    success = fix_common_py()
    sys.exit(0 if success else 1)
