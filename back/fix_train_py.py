#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script autonome pour corriger le problème d'autocast dans train.py
A exécuter directement si l'erreur persiste
"""

import os
import re
import sys
import argparse

def fix_autocast_problem(train_path):
    """
    Corrige le problème spécifique d'autocast dans train.py
    
    Args:
        train_path (str): Chemin complet vers le fichier train.py
    """
    print(f"\n=== Correction du problème d'autocast dans {train_path} ===")
    
    if not os.path.exists(train_path):
        print(f"❌ Fichier {train_path} introuvable!")
        return False
    
    try:
        # Lire le contenu du fichier
        with open(train_path, 'r') as f:
            content = f.read()
        
        # Chercher le pattern problématique et le remplacer
        if "with amp.autocast(" in content:
            # Version 1 - Remplacer par la solution la plus simple
            new_content = re.sub(
                r"with amp\.autocast\(['\"]cuda['\"]\s*,\s*enabled=(\w+)\):",
                r"with amp.autocast(enabled=\1):",
                content
            )
            
            # Si le contenu a été modifié
            if new_content != content:
                # Sauvegarder le fichier modifié
                with open(train_path, 'w') as f:
                    f.write(new_content)
                
                print("✅ Problème d'autocast corrigé avec succès!")
                return True
            else:
                # Essayer d'autres patterns
                new_content = re.sub(
                    r"with amp\.autocast\(device_type=['\"]cuda['\"]\s*,\s*enabled=(\w+)\):",
                    r"with amp.autocast(enabled=\1):",
                    content
                )
                
                if new_content != content:
                    # Sauvegarder le fichier modifié
                    with open(train_path, 'w') as f:
                        f.write(new_content)
                    
                    print("✅ Problème d'autocast avec device_type corrigé avec succès!")
                    return True
                else:
                    print("⚠️ Aucun pattern connu d'autocast trouvé dans le fichier.")
                    
                    # Rechercher et afficher les lignes avec autocast pour diagnostic
                    lines = content.split('\n')
                    autocast_lines = []
                    for i, line in enumerate(lines):
                        if "amp.autocast" in line:
                            autocast_lines.append((i+1, line.strip()))
                    
                    if autocast_lines:
                        print("\nLignes avec autocast trouvées:")
                        for line_num, line_content in autocast_lines:
                            print(f"  Ligne {line_num}: {line_content}")
                        
                        # Tentative de correction manuelle basée sur le contenu trouvé
                        if len(autocast_lines) == 1:
                            _, line_content = autocast_lines[0]
                            corrected_line = re.sub(
                                r"amp\.autocast\(.*\)",
                                r"amp.autocast(enabled=cuda)",
                                line_content
                            )
                            
                            if corrected_line != line_content:
                                print(f"\nAppliquation de la correction manuelle:")
                                print(f"  Ancien: {line_content}")
                                print(f"  Nouveau: {corrected_line}")
                                
                                # Remplacer la ligne dans le contenu
                                new_content = content.replace(line_content, corrected_line)
                                
                                # Sauvegarder le fichier modifié
                                with open(train_path, 'w') as f:
                                    f.write(new_content)
                                
                                print("✅ Correction manuelle appliquée avec succès!")
                                return True
                    
                    return False
        else:
            print("⚠️ Aucun appel à amp.autocast trouvé dans le fichier.")
            return False
    except Exception as e:
        print(f"❌ Erreur lors de la correction: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Corriger le problème autocast dans train.py')
    parser.add_argument('--path', type=str, default='/content/yolov5-face/train.py',
                        help='Chemin vers le fichier train.py')
    
    args = parser.parse_args()
    
    # Vérifier si le chemin est valide
    if not os.path.exists(args.path):
        # Essayer de trouver le fichier automatiquement
        possible_paths = [
            '/content/yolov5-face/train.py',
            os.path.join(os.getcwd(), 'yolov5-face', 'train.py'),
            os.path.join(os.path.dirname(os.getcwd()), 'yolov5-face', 'train.py')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Fichier train.py trouvé automatiquement: {path}")
                args.path = path
                break
    
    if os.path.exists(args.path):
        fix_autocast_problem(args.path)
    else:
        print(f"❌ Impossible de trouver le fichier train.py. Veuillez spécifier le chemin correct avec --path.")

if __name__ == "__main__":
    main()
