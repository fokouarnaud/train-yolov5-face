#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script définitif pour corriger le problème de conversion de type dans loss.py
Ce script utilise plusieurs approches pour garantir que la correction est appliquée correctement
"""

import os
import re
import traceback
import subprocess

def fix_loss_py(yolo_dir='/content/yolov5-face'):
    """
    Corrige définitivement le problème de conversion de type dans le fichier loss.py
    
    Args:
        yolo_dir (str): Répertoire de YOLOv5-Face
        
    Returns:
        bool: True si la correction a réussi, False sinon
    """
    print("\n" + "=" * 70)
    print(" CORRECTION DÉFINITIVE DU FICHIER LOSS.PY ".center(70, "="))
    print("=" * 70 + "\n")
    
    loss_py_path = os.path.join(yolo_dir, 'utils', 'loss.py')
    
    if not os.path.exists(loss_py_path):
        print(f"❌ ERREUR: Fichier {loss_py_path} non trouvé!")
        print("Assurez-vous que le dépôt YOLOv5-Face a bien été cloné.")
        return False
    
    # Sauvegarder une copie du fichier original
    backup_path = loss_py_path + '.backup'
    try:
        with open(loss_py_path, 'r') as f_src:
            with open(backup_path, 'w') as f_dst:
                f_dst.write(f_src.read())
        print(f"✓ Sauvegarde créée: {backup_path}")
    except Exception as e:
        print(f"⚠️ Impossible de créer une sauvegarde: {e}")
    
    try:
        # MÉTHODE 1: Utilisation de commandes sed (plus fiable)
        try:
            print("Méthode 1: Utilisation de sed pour remplacer les expressions clamp_...")
            
            # Remplacer les deux expressions clamp_ en une seule commande
            subprocess.run([
                'sed', '-i', 
                's/gj.clamp_\\(0, gain\\[3\\] - 1\\)/gj.clamp_(0, gain[3] - 1).long()/g; s/gi.clamp_\\(0, gain\\[2\\] - 1\\)/gi.clamp_(0, gain[2] - 1).long()/g', 
                loss_py_path
            ], check=False)
            
            # Vérifier si la modification a été appliquée
            with open(loss_py_path, 'r') as f:
                content = f.read()
            
            if '.long()' in content:
                print("✅ Méthode 1 réussie: Modifications appliquées avec sed")
                return True
            else:
                print("ℹ️ Méthode 1 n'a pas appliqué les modifications. Essai avec la méthode 2...")
        except Exception as e:
            print(f"ℹ️ Méthode 1 a échoué: {e}")
        
        # MÉTHODE 2: Recherche du motif exact et remplacement
        print("Méthode 2: Recherche du motif exact et remplacement...")
        
        with open(loss_py_path, 'r') as f:
            content = f.read()
        
        exact_pattern = 'indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))'
        replacement = 'indices.append((b, a, gj.clamp_(0, gain[3] - 1).long(), gi.clamp_(0, gain[2] - 1).long()))'
        
        if exact_pattern in content:
            modified_content = content.replace(exact_pattern, replacement)
            
            with open(loss_py_path, 'w') as f:
                f.write(modified_content)
            
            print("✅ Méthode 2 réussie: Motif exact trouvé et remplacé")
            return True
        else:
            print("ℹ️ Motif exact non trouvé. Essai avec la méthode 3...")
        
        # MÉTHODE 3: Utilisation d'expressions régulières plus flexibles
        print("Méthode 3: Utilisation d'expressions régulières...")
        
        with open(loss_py_path, 'r') as f:
            lines = f.readlines()
        
        regex_pattern = r'indices\.append\(\(b,\s*a,\s*gj\.clamp_\(0,\s*gain\[3\]\s*-\s*1\)(\.long\(\))?,\s*gi\.clamp_\(0,\s*gain\[2\]\s*-\s*1\)(\.long\(\))?\)\)'
        
        found = False
        for i, line in enumerate(lines):
            match = re.search(regex_pattern, line)
            if match:
                # Vérifier si .long() est déjà présent
                has_long_gj = match.group(1) is not None
                has_long_gi = match.group(2) is not None
                
                if not has_long_gj or not has_long_gi:
                    # Construire la nouvelle ligne avec .long()
                    # Format: indices.append((b, a, gj.clamp_(0, gain[3] - 1).long(), gi.clamp_(0, gain[2] - 1).long()))
                    parts = re.split(r'(gj\.clamp_\(0,\s*gain\[3\]\s*-\s*1\))', line)
                    if len(parts) >= 3:
                        parts[2] = '.long()' + parts[2] if not has_long_gj else parts[2]
                    
                    parts = re.split(r'(gi\.clamp_\(0,\s*gain\[2\]\s*-\s*1\))', ''.join(parts))
                    if len(parts) >= 3:
                        parts[2] = '.long()' + parts[2] if not has_long_gi else parts[2]
                    
                    new_line = ''.join(parts)
                    lines[i] = new_line
                    
                    print(f"✅ Ligne {i+1} modifiée avec regex:")
                    print(f"  Avant: {line.strip()}")
                    print(f"  Après: {new_line.strip()}")
                    found = True
                else:
                    print(f"ℹ️ Ligne {i+1} déjà modifiée (contient .long())")
                    found = True
        
        if found:
            with open(loss_py_path, 'w') as f:
                f.writelines(lines)
            
            print("✅ Méthode 3 réussie: Modifications appliquées avec regex")
            return True
        else:
            print("ℹ️ Motif regex non trouvé. Essai avec la méthode 4...")
        
        # MÉTHODE 4: Analyse ligne par ligne pour trouver les lignes contenant clamp_
        print("Méthode 4: Analyse ligne par ligne...")
        
        with open(loss_py_path, 'r') as f:
            lines = f.readlines()
        
        # Rechercher les lignes qui contiennent à la fois "indices.append" et "clamp_"
        found = False
        for i, line in enumerate(lines):
            if 'indices.append' in line and 'clamp_' in line and '.long()' not in line:
                print(f"Ligne trouvée ({i+1}): {line.strip()}")
                
                # Remplacer les clamp_ par clamp_().long()
                modified_line = re.sub(
                    r'(gj|gi)\.clamp_\(([^)]+)\)',
                    r'\1.clamp_(\2).long()',
                    line
                )
                
                lines[i] = modified_line
                print(f"✅ Ligne modifiée: {modified_line.strip()}")
                found = True
        
        if found:
            with open(loss_py_path, 'w') as f:
                f.writelines(lines)
            
            print("✅ Méthode 4 réussie: Modifications appliquées ligne par ligne")
            return True
        else:
            print("ℹ️ Aucune ligne compatible trouvée. Essai avec la méthode 5...")
        
        # MÉTHODE 5: Modification directe basée sur les numéros de ligne connus
        print("Méthode 5: Modification basée sur les numéros de ligne connus...")
        
        # Selon l'erreur, le problème est à la ligne 248 de loss.py
        target_line_numbers = [248, 249, 247, 250]  # Essayer plusieurs lignes autour de 248
        
        with open(loss_py_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < max(target_line_numbers):
            print(f"⚠️ Le fichier ne contient que {len(lines)} lignes, mais nous ciblons la ligne {max(target_line_numbers)}")
        
        found = False
        for line_num in target_line_numbers:
            if line_num <= len(lines):
                line_index = line_num - 1  # Ajustement pour l'indexation Python (0-based)
                line = lines[line_index]
                
                if 'clamp_' in line and 'indices.append' in line and '.long()' not in line:
                    print(f"Ligne cible trouvée ({line_num}): {line.strip()}")
                    
                    # Utiliser une expression régulière pour insérer .long() après chaque appel clamp_()
                    modified_line = re.sub(
                        r'(\.clamp_\([^)]+\))',
                        r'\1.long()',
                        line
                    )
                    
                    lines[line_index] = modified_line
                    print(f"✅ Ligne modifiée: {modified_line.strip()}")
                    found = True
                    break
        
        if found:
            with open(loss_py_path, 'w') as f:
                f.writelines(lines)
            
            print("✅ Méthode 5 réussie: Modification appliquée à la ligne cible")
            return True
        else:
            print("❌ Aucune méthode n'a réussi à appliquer la correction.")
            
            # Restaurer la sauvegarde si elle existe
            if os.path.exists(backup_path):
                try:
                    with open(backup_path, 'r') as f_src:
                        with open(loss_py_path, 'w') as f_dst:
                            f_dst.write(f_src.read())
                    print("✓ Fichier original restauré depuis la sauvegarde")
                except Exception as e:
                    print(f"⚠️ Erreur lors de la restauration: {e}")
            
            # Afficher les lignes contenant clamp_ pour aider à un correctif manuel
            print("\n🔍 Voici toutes les lignes contenant 'clamp_' dans le fichier:")
            with open(loss_py_path, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if 'clamp_' in line:
                        print(f"Ligne {i+1}: {line.strip()}")
            
            print("\n⚠️ CORRECTION MANUELLE NÉCESSAIRE:")
            print("1. Localisez la ligne qui contient à la fois 'indices.append' et 'clamp_'")
            print("2. Modifiez cette ligne en ajoutant '.long()' après chaque 'clamp_(...)'")
            print("3. Par exemple, transformez:")
            print("   indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))")
            print("   en:")
            print("   indices.append((b, a, gj.clamp_(0, gain[3] - 1).long(), gi.clamp_(0, gain[2] - 1).long()))")
            
            return False
            
    except Exception as e:
        print(f"\n❌ Erreur lors de la correction de loss.py: {e}")
        print(traceback.format_exc())
        
        # Restaurer la sauvegarde si elle existe
        if os.path.exists(backup_path):
            try:
                with open(backup_path, 'r') as f_src:
                    with open(loss_py_path, 'w') as f_dst:
                        f_dst.write(f_src.read())
                print("✓ Fichier original restauré depuis la sauvegarde")
            except Exception as e:
                print(f"⚠️ Erreur lors de la restauration: {e}")
        
        return False
    
    # Vérification finale
    with open(loss_py_path, 'r') as f:
        content = f.read()
    
    if '.long()' in content:
        # Supprimer la sauvegarde si tout s'est bien passé
        if os.path.exists(backup_path):
            try:
                os.remove(backup_path)
                print("✓ Sauvegarde supprimée")
            except Exception as e:
                print(f"⚠️ Impossible de supprimer la sauvegarde: {e}")
        
        print("\n🎉 SUCCÈS: Le fichier loss.py a été correctement modifié!")
        print("Vous pouvez maintenant reprendre l'entraînement.")
        return True
    else:
        print("\n❌ ÉCHEC: Le fichier loss.py n'a pas été correctement modifié!")
        return False

if __name__ == "__main__":
    fix_loss_py()
