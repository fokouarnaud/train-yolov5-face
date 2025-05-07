#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script pour patcher train.py du dépôt YOLOv5-Face
afin de le rendre compatible avec PyTorch 2.6+
"""

import os
import re
import argparse

def patch_train_script(yolo_dir):
    """
    Modifie le script train.py pour supporter PyTorch 2.6+
    en ajoutant le paramètre weights_only=False à torch.load()
    
    Args:
        yolo_dir (str): Répertoire du dépôt YOLOv5-Face
    """
    train_path = os.path.join(yolo_dir, 'train.py')
    
    if not os.path.exists(train_path):
        print(f"✗ Fichier train.py non trouvé: {train_path}")
        return False
    
    # Lire le contenu du fichier
    with open(train_path, 'r') as f:
        content = f.read()
    
    # Chercher diverses formes de torch.load sans weights_only
    patterns = [
        r'(torch\.load\(weights,\s*map_location=device\))',  # Forme la plus courante
        r'(torch\.load\(weights,\s*map_location\s*=\s*device\))',  # Avec des espaces variables
        r'(ckpt\s*=\s*torch\.load\(weights,\s*map_location\s*=\s*device\))'  # Avec assignation
    ]
    
    # Vérifier si l'un des patterns est trouvé
    pattern_found = False
    new_content = content
    
    for pattern in patterns:
        if re.search(pattern, new_content):
            pattern_found = True
            replacement = r'\1, weights_only=False'
            new_content = re.sub(pattern, replacement, new_content)
    
    if not pattern_found:
        # Essayer une approche plus simple si les regex échouent
        if 'torch.load(weights, map_location=device)' in new_content:
            new_content = new_content.replace(
                'torch.load(weights, map_location=device)',
                'torch.load(weights, map_location=device, weights_only=False)'
            )
            pattern_found = True
    
    if not pattern_found:
        print("✗ Motif de code non trouvé dans train.py. Le script a peut-être changé.")
        return False
    
    # Sauvegarder le fichier modifié
    with open(train_path, 'w') as f:
        f.write(new_content)
    
    print(f"✓ Fichier train.py modifié avec succès: {train_path}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patcher train.py pour PyTorch 2.6+")
    parser.add_argument('--yolo-dir', type=str, default='/content/yolov5-face',
                        help='Chemin du dépôt YOLOv5-Face')
    
    args = parser.parse_args()
    patch_train_script(args.yolo_dir)
