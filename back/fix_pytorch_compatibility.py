#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script pour corriger la compatibilité avec PyTorch 2.6+
Ce script modifie directement le fichier train.py du dépôt YOLOv5-Face
pour ajouter le paramètre weights_only=False à torch.load()
"""

import os
import re
import sys
import subprocess

def fix_train_script(yolo_dir='/content/yolov5-face'):
    """
    Corrige le fichier train.py pour la compatibilité avec PyTorch 2.6+
    
    Args:
        yolo_dir (str): Chemin du dépôt YOLOv5-Face
        
    Returns:
        bool: True si la correction a réussi, False sinon
    """
    print("=" * 80)
    print("CORRECTION DE COMPATIBILITÉ PYTORCH 2.6+")
    print("=" * 80)
    
    train_path = os.path.join(yolo_dir, 'train.py')
    
    if not os.path.exists(train_path):
        print(f"✗ ERREUR: Fichier train.py non trouvé: {train_path}")
        print(f"  Vérifiez que le dépôt YOLOv5-Face a bien été cloné.")
        return False
    
    print(f"Fichier train.py trouvé: {train_path}")
    
    # Lire le contenu original du fichier
    with open(train_path, 'r') as f:
        original_content = f.read()
    
    print("\nRecherche de motifs de torch.load()...")
    
    # Utiliser des expressions régulières pour trouver toutes les occurrences de torch.load
    # avec weights et map_location mais sans weights_only
    torch_load_patterns = [
        # Motif de base
        (r'torch\.load\(weights,\s*map_location=device\)', 
         r'torch.load(weights, map_location=device, weights_only=False)'),
        
        # Motif avec assignation à ckpt
        (r'ckpt\s*=\s*torch\.load\(weights,\s*map_location=device\)', 
         r'ckpt = torch.load(weights, map_location=device, weights_only=False)'),
        
        # Motif sans espace après la virgule
        (r'torch\.load\(weights,map_location=device\)', 
         r'torch.load(weights,map_location=device, weights_only=False)'),
        
        # Motif avec espaces variables
        (r'torch\.load\(\s*weights\s*,\s*map_location\s*=\s*device\s*\)', 
         r'torch.load(weights, map_location=device, weights_only=False)'),
    ]
    
    # Appliquer tous les motifs et vérifier si des changements ont été effectués
    modified_content = original_content
    changes_made = False
    
    for pattern, replacement in torch_load_patterns:
        # Vérifier si le motif existe et n'a pas déjà été corrigé
        matches = re.findall(pattern, modified_content)
        if matches and 'weights_only=False' not in ''.join(matches):
            print(f"✓ Motif trouvé: {matches[0]}")
            modified_content = re.sub(pattern, replacement, modified_content)
            changes_made = True
    
    # Si aucun motif n'a été trouvé avec les regex, essayer une approche directe
    if not changes_made:
        direct_patterns = [
            ('torch.load(weights, map_location=device)', 'torch.load(weights, map_location=device, weights_only=False)'),
            ('ckpt = torch.load(weights, map_location=device)', 'ckpt = torch.load(weights, map_location=device, weights_only=False)'),
        ]
        
        for old_str, new_str in direct_patterns:
            if old_str in modified_content and new_str not in modified_content:
                modified_content = modified_content.replace(old_str, new_str)
                print(f"✓ Remplacement direct: {old_str}")
                changes_made = True
    
    # Dernière tentative: chercher les lignes contenant torch.load et weights
    if not changes_made:
        print("\nAucun motif standard trouvé. Recherche de lignes contenant torch.load()...")
        
        lines = original_content.split('\n')
        for i, line in enumerate(lines):
            if 'torch.load' in line and 'weights' in line and 'map_location' in line and 'weights_only=False' not in line:
                print(f"Ligne trouvée ({i+1}): {line}")
                
                # Vérifier où insérer weights_only=False
                if ')' in line:
                    modified_line = line.replace(')', ', weights_only=False)', 1)
                    lines[i] = modified_line
                    print(f"Ligne modifiée: {modified_line}")
                    changes_made = True
        
        if changes_made:
            modified_content = '\n'.join(lines)
    
    # Vérifier si des modifications ont été apportées
    if original_content == modified_content:
        print("\n⚠️ ATTENTION: Aucune modification n'a été effectuée!")
        print("Le fichier train.py ne contient peut-être pas le format attendu de torch.load().")
        
        # Afficher toutes les lignes contenant torch.load pour diagnostic
        print("\nLignes contenant torch.load dans le fichier original:")
        for i, line in enumerate(original_content.split('\n')):
            if 'torch.load' in line:
                print(f"Ligne {i+1}: {line}")
        
        return False
    
    # Sauvegarder le fichier modifié
    with open(train_path, 'w') as f:
        f.write(modified_content)
    
    # Vérifier que le fichier a bien été modifié
    with open(train_path, 'r') as f:
        final_content = f.read()
    
    if 'weights_only=False' in final_content:
        print("\n✅ SUCCÈS: Le fichier train.py a été correctement modifié!")
        print("Le paramètre weights_only=False a été ajouté à la fonction torch.load().")
        return True
    else:
        print("\n❌ ÉCHEC: La modification n'a pas été correctement sauvegardée!")
        return False

def main():
    """Point d'entrée du script"""
    # Utiliser le chemin spécifié en argument ou la valeur par défaut
    yolo_dir = sys.argv[1] if len(sys.argv) > 1 else '/content/yolov5-face'
    
    # Vérifier si le dépôt existe, sinon le cloner
    if not os.path.exists(yolo_dir):
        print(f"Le dépôt YOLOv5-Face n'existe pas à l'emplacement {yolo_dir}")
        print("Clonage du dépôt...")
        subprocess.run(['git', 'clone', 'https://github.com/deepcam-cn/yolov5-face.git', yolo_dir], check=True)
    
    # Appliquer la correction
    success = fix_train_script(yolo_dir)
    
    if success:
        print("\nVous pouvez maintenant lancer l'entraînement avec la commande:")
        print("python main.py")
    else:
        print("\nVous devez corriger manuellement le fichier train.py avant de lancer l'entraînement.")
        print("Recherchez la ligne contenant 'torch.load(weights, map_location=device)'")
        print("et remplacez-la par 'torch.load(weights, map_location=device, weights_only=False)'")

if __name__ == "__main__":
    main()
