#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script amélioré pour corriger l'utilisation de amp.autocast et amp.GradScaler dans train.py
Compatible avec plusieurs versions de PyTorch
"""

import os
import re

def fix_autocast_api(yolo_dir=None):
    """
    Met à jour les appels à autocast et GradScaler pour les rendre compatibles
    avec les versions récentes de PyTorch
    
    Args:
        yolo_dir (str, optional): Répertoire de YOLOv5-face. Si None, utilise '/content/yolov5-face'.
    """
    if yolo_dir is None:
        train_path = '/content/yolov5-face/train.py'
    else:
        train_path = os.path.join(yolo_dir, 'train.py')
    
    print(f"\n=== Correction des API PyTorch pour {train_path} ===")
    
    if not os.path.exists(train_path):
        print(f"❌ Erreur: Le fichier {train_path} n'existe pas!")
        return False
    
    try:
        with open(train_path, 'r') as f:
            content = f.read()
        
        # Tester les différentes formes possibles de l'appel à autocast
        # Solution 1 - La plus simple (souvent la plus compatible)
        if "amp.autocast('cuda', enabled=cuda)" in content:
            content = content.replace(
                "amp.autocast('cuda', enabled=cuda)",
                "amp.autocast(enabled=cuda)"
            )
            modified = True
            print("✅ Solution 1 appliquée pour autocast")
        # Solution 2 - Avec device_type
        elif re.search(r"amp\.autocast\(['\"](cuda|cpu)['\"]\s*,\s*enabled=(?:cuda|True|False)\)", content):
            content = re.sub(
                r"amp\.autocast\(['\"](cuda|cpu)['\"]\s*,\s*enabled=(cuda|True|False)\)",
                r"amp.autocast(enabled=\2)",
                content
            )
            modified = True
            print("✅ Solution 2 appliquée pour autocast")
        # Solution 3 - Autre pattern
        else:
            # Essayer d'autres patterns possibles
            original = content
            content = re.sub(
                r"amp\.autocast\(['\"](cuda|cpu)['\"]\s*,\s*enabled=(\w+)\)",
                r"amp.autocast(enabled=\2)",
                content
            )
            modified = (original != content)
            if modified:
                print("✅ Solution 3 appliquée pour autocast")
            else:
                print("⚠️ Aucun pattern d'autocast trouvé à corriger")
        
        # Corriger aussi GradScaler si nécessaire
        if "amp.GradScaler(" in content:
            original = content
            content = re.sub(
                r"amp\.GradScaler\(['\"](cuda|cpu)['\"]\s*,\s*enabled=(\w+)\)",
                r"amp.GradScaler(enabled=\2)",
                content
            )
            grad_scaler_modified = (original != content)
            if grad_scaler_modified:
                print("✅ GradScaler également corrigé")
        
        # Écrire les modifications dans le fichier
        with open(train_path, 'w') as f:
            f.write(content)
        
        print("✅ Modifications appliquées avec succès au fichier train.py")
        return True
    except Exception as e:
        print(f"❌ Erreur lors de la correction des API PyTorch: {str(e)}")
        return False

if __name__ == "__main__":
    fix_autocast_api()
