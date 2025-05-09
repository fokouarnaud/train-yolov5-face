#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script pour corriger le problème de la virgule dans train.py
"""

import os
import re

def fix_train_script():
    """
    Corrige le problème de la virgule dans train.py
    """
    train_path = '/content/yolov5-face/train.py'
    
    if not os.path.exists(train_path):
        print(f"❌ Erreur: Le fichier {train_path} n'existe pas!")
        return False
    
    # Lire le contenu du fichier
    with open(train_path, 'r') as f:
        content = f.read()
    
    # Chercher le motif problématique avec une virgule à la fin
    pattern = r'(ckpt\s*=\s*torch\.load\s*\(\s*weights\s*,\s*map_location\s*=\s*device\s*\))\s*,'
    
    if re.search(pattern, content):
        # Remplacer le motif avec la virgule par le motif corrigé sans virgule
        modified_content = re.sub(
            pattern,
            'ckpt = torch.load(weights, map_location=device, weights_only=False)',
            content
        )
        
        # Écrire le contenu modifié
        with open(train_path, 'w') as f:
            f.write(modified_content)
        
        print("✅ Correction appliquée avec succès! La virgule problématique a été supprimée.")
        return True
    else:
        # Si le motif avec la virgule n'est pas trouvé, vérifier le motif sans virgule
        normal_pattern = r'ckpt\s*=\s*torch\.load\s*\(\s*weights\s*,\s*map_location\s*=\s*device\s*\)'
        
        if re.search(normal_pattern, content):
            # Remplacer le motif normal par le motif avec weights_only=False
            modified_content = re.sub(
                normal_pattern,
                'ckpt = torch.load(weights, map_location=device, weights_only=False)',
                content
            )
            
            # Écrire le contenu modifié
            with open(train_path, 'w') as f:
                f.write(modified_content)
            
            print("✅ Le motif normal a été trouvé et corrigé en ajoutant weights_only=False.")
            return True
        else:
            print("❌ Aucun motif compatible n'a été trouvé dans le fichier.")
            
            # Chercher toutes les occurrences de torch.load pour aider au diagnostic
            torch_load_lines = []
            for i, line in enumerate(content.split('\n')):
                if 'torch.load' in line:
                    torch_load_lines.append(f"Ligne {i+1}: {line}")
            
            if torch_load_lines:
                print("\nLignes contenant 'torch.load' trouvées:")
                for line in torch_load_lines:
                    print(line)
            
            return False

if __name__ == "__main__":
    fix_train_script()
