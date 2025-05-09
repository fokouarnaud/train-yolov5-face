#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script pour corriger le fichier train.py de YOLOv5-Face
Ce script doit être exécuté juste APRÈS le script colab_setup.py
et AVANT le lancement de l'entraînement avec main.py

Ce script est conçu pour être robuste contre les exécutions multiples
et corrige également les applications incorrectes précédentes.
"""

import os
import re
import sys

def fix_pytorch_compatibility(yolo_dir='/content/yolov5-face'):
    """
    Corrige le fichier train.py pour PyTorch 2.6+
    en ajoutant correctement le paramètre weights_only=False à torch.load
    """
    print("\n" + "=" * 70)
    print(" CORRECTION DU FICHIER TRAIN.PY POUR PYTORCH 2.6+ ".center(70, "="))
    print("=" * 70 + "\n")
    
    train_path = '/content/yolov5-face/train.py'
    
    if not os.path.exists(train_path):
        print(f"❌ ERREUR: Fichier {train_path} non trouvé!")
        print("Assurez-vous que le dépôt YOLOv5-Face a bien été cloné.")
        return False
    
    # Lire le contenu du fichier
    with open(train_path, 'r') as f:
        content = f.read()
    
    # ÉTAPE 1: Vérifier si le fichier contient déjà une correction incorrecte
    # et la corriger en premier lieu
    incorrect_patterns = [
        r'(ckpt\s*=\s*torch\.load\s*\(\s*weights\s*,\s*map_location\s*=\s*device\s*\))\s*,\s*weights_only\s*=\s*False',
        r'(torch\.load\s*\(\s*weights\s*,\s*map_location\s*=\s*device\s*\))\s*,\s*weights_only\s*=\s*False',
    ]
    
    for pattern in incorrect_patterns:
        match = re.search(pattern, content)
        if match:
            # Trouvé une correction incorrecte (paramètre outside des parenthèses)
            full_match = match.group(0)
            inner_part = match.group(1)
            
            # Créer la correction appropriée
            correct_replacement = inner_part.replace(')', ', weights_only=False)')
            
            # Remplacer la partie incorrecte par la version corrigée
            content = content.replace(full_match, correct_replacement)
            print(f"✅ Correction d'une application incorrecte précédente: {full_match}")
            print(f"   Nouvelle version: {correct_replacement}")
    
    # ÉTAPE 2: Corriger les cas où le paramètre a été ajouté en double, triple, etc.
    # Exemple: ckpt = torch.load(weights, map_location=device, weights_only=False, weights_only=False)
    doubled_pattern = r'torch\.load\s*\([^)]*weights_only\s*=\s*False[^)]*weights_only\s*=\s*False'
    
    if re.search(doubled_pattern, content):
        print("⚠️ Détection d'une multiple application du paramètre weights_only=False")
        
        # Extraire toutes les lignes avec torch.load
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'torch.load' in line and 'weights' in line:
                # Comptez combien de fois 'weights_only=False' apparaît
                weights_only_count = line.count('weights_only=False')
                
                if weights_only_count > 1:
                    print(f"Ligne problématique trouvée ({i+1}): {line}")
                    
                    # Construire une nouvelle ligne correcte
                    if 'ckpt =' in line:
                        # Cas pour ckpt = torch.load(...)
                        new_line = 'ckpt = torch.load(weights, map_location=device, weights_only=False)  # load checkpoint'
                    else:
                        # Autre cas général
                        new_line = line.split('torch.load')[0] + 'torch.load(weights, map_location=device, weights_only=False)'
                        if '#' in line:
                            new_line += '  ' + line.split('#')[-1]
                    
                    lines[i] = new_line
                    print(f"✅ Ligne corrigée: {new_line}")
        
        # Reconstruire le contenu
        content = '\n'.join(lines)
    
    # ÉTAPE 3: Vérifier si le paramètre weights_only=False est correctement présent
    # S'il est déjà présent et correctement formaté, pas besoin d'appliquer la correction
    correct_pattern = r'torch\.load\s*\(\s*weights\s*,\s*map_location\s*=\s*device\s*,\s*weights_only\s*=\s*False\s*\)'
    
    if re.search(correct_pattern, content):
        print("✅ Le fichier train.py est déjà correctement modifié.")
        print("Le paramètre weights_only=False est présent au bon format.")
        
        # Sauvegarder le contenu au cas où il a été modifié à l'étape 1 ou 2
        with open(train_path, 'w') as f:
            f.write(content)
        
        return True
    
    # ÉTAPE 4: Appliquer la correction si nécessaire
    print("🔄 Application de la correction...")
    
    # Chercher la ligne qui correspond au format standard
    standard_pattern = r'ckpt\s*=\s*torch\.load\s*\(\s*weights\s*,\s*map_location\s*=\s*device\s*\)'
    
    if re.search(standard_pattern, content):
        # Trouvé le format standard
        match = re.search(standard_pattern, content)
        original = match.group(0)
        replacement = original.replace(')', ', weights_only=False)')
        
        content = content.replace(original, replacement)
        print(f"✅ Correction appliquée: {original} → {replacement}")
    else:
        # Si le format standard n'est pas trouvé, rechercher d'autres variantes
        print("🔍 Format standard non trouvé, recherche d'autres variantes...")
        
        # Chercher toutes les lignes qui contiennent torch.load avec weights et map_location
        lines = content.split('\n')
        found = False
        
        for i, line in enumerate(lines):
            if 'torch.load' in line and 'weights' in line and 'map_location' in line and 'weights_only=False' not in line:
                print(f"Ligne trouvée ({i+1}): {line}")
                
                # Trouver la position de la parenthèse fermante après torch.load
                open_paren = line.find('torch.load(')
                if open_paren >= 0:
                    # Compter les parenthèses ouvrantes et fermantes pour trouver la bonne parenthèse fermante
                    open_count = 0
                    close_paren_pos = -1
                    
                    for j, char in enumerate(line[open_paren:]):
                        if char == '(':
                            open_count += 1
                        elif char == ')':
                            open_count -= 1
                            if open_count == 0:
                                close_paren_pos = open_paren + j
                                break
                    
                    if close_paren_pos > 0:
                        # Insérer weights_only=False juste avant la parenthèse fermante
                        modified_line = line[:close_paren_pos] + ', weights_only=False' + line[close_paren_pos:]
                        lines[i] = modified_line
                        print(f"✅ Ligne modifiée: {modified_line}")
                        found = True
                        break
        
        if found:
            content = '\n'.join(lines)
        else:
            print("❌ Impossible de trouver une ligne appropriée pour appliquer la correction.")
            return False
    
    # Sauvegarder le fichier modifié
    try:
        with open(train_path, 'w') as f:
            f.write(content)
        
        # Vérifier que la modification a bien été appliquée
        with open(train_path, 'r') as f:
            check_content = f.read()
        
        if 'weights_only=False' in check_content:
            if re.search(correct_pattern, check_content) or not re.search(doubled_pattern, check_content):
                print("\n✅ SUCCÈS: Le fichier train.py a été correctement modifié!")
                print("Le paramètre weights_only=False a bien été ajouté au bon format.")
                return True
            else:
                print("\n⚠️ La correction a été appliquée mais pourrait nécessiter une vérification.")
                return True
        else:
            print("\n❌ ERREUR: La modification n'a pas été correctement enregistrée!")
            return False
            
    except Exception as e:
        print(f"\n❌ ERREUR lors de la sauvegarde du fichier: {e}")
        return False

if __name__ == "__main__":
    fix_pytorch_compatibility()
