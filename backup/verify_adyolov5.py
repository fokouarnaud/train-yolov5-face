#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script pour vérifier la présence des fichiers ADYOLOv5-Face
et pour empêcher leur génération automatique
"""

import os
import sys
import shutil
import argparse

def verify_adyolov5_files(yolo_dir, backup_dir=None):
    """
    Vérifie que les fichiers ADYOLOv5 sont présents dans le dépôt
    Si backup_dir est spécifié, sauvegarde les fichiers générés automatiquement
    et les remplace par les fichiers personnalisés depuis le backup
    """
    # Chemins des fichiers ADYOLOv5
    gd_path = os.path.join(yolo_dir, 'models', 'gd.py')
    adyolo_yaml_path = os.path.join(yolo_dir, 'models', 'adyolov5s_simple.yaml')
    hyp_adyolo_path = os.path.join(yolo_dir, 'data', 'hyp.adyolo.yaml')
    
    # Vérifier si les fichiers existent
    files_exist = [
        os.path.exists(gd_path),
        os.path.exists(adyolo_yaml_path),
        os.path.exists(hyp_adyolo_path)
    ]
    
    print("\n=== Vérification des fichiers ADYOLOv5-Face ===")
    
    if all(files_exist):
        print("✓ Tous les fichiers ADYOLOv5-Face sont présents dans le dépôt:")
        print(f"  - {gd_path}")
        print(f"  - {adyolo_yaml_path}")
        print(f"  - {hyp_adyolo_path}")
    else:
        print("⚠️ ATTENTION: Certains fichiers ADYOLOv5-Face sont manquants:")
        if not files_exist[0]:
            print(f"  ✗ Fichier manquant: {gd_path}")
        if not files_exist[1]:
            print(f"  ✗ Fichier manquant: {adyolo_yaml_path}")
        if not files_exist[2]:
            print(f"  ✗ Fichier manquant: {hyp_adyolo_path}")
        print("")
        
        if backup_dir and os.path.exists(backup_dir):
            print(f"Tentative de restauration des fichiers manquants depuis {backup_dir}...")
            
            # Restaurer les fichiers manquants depuis le répertoire de sauvegarde
            backup_gd = os.path.join(backup_dir, 'gd.py')
            backup_yaml = os.path.join(backup_dir, 'adyolov5s_simple.yaml')
            backup_hyp = os.path.join(backup_dir, 'hyp.adyolo.yaml')
            
            if not files_exist[0] and os.path.exists(backup_gd):
                os.makedirs(os.path.dirname(gd_path), exist_ok=True)
                shutil.copy(backup_gd, gd_path)
                print(f"  ✓ Restauré: {gd_path}")
            
            if not files_exist[1] and os.path.exists(backup_yaml):
                os.makedirs(os.path.dirname(adyolo_yaml_path), exist_ok=True)
                shutil.copy(backup_yaml, adyolo_yaml_path)
                print(f"  ✓ Restauré: {adyolo_yaml_path}")
            
            if not files_exist[2] and os.path.exists(backup_hyp):
                os.makedirs(os.path.dirname(hyp_adyolo_path), exist_ok=True)
                shutil.copy(backup_hyp, hyp_adyolo_path)
                print(f"  ✓ Restauré: {hyp_adyolo_path}")
        else:
            print("Ces fichiers doivent être présents dans le dépôt GitHub.")
            print("Vérifiez que vous avez bien poussé vos modifications vers GitHub.")
    
    # Vérifier le support du mécanisme GD dans yolo.py
    yolo_py_path = os.path.join(yolo_dir, 'models', 'yolo.py')
    if os.path.exists(yolo_py_path):
        with open(yolo_py_path, 'r') as f:
            yolo_content = f.read()
        
        if 'from models.gd import' in yolo_content:
            print("\n✓ Le fichier yolo.py contient le support du mécanisme GD")
        else:
            print("\n⚠️ Le fichier yolo.py ne contient pas les imports pour le mécanisme GD")
            print("  Il est possible que la modification de yolo.py n'ait pas été effectuée correctement.")
            
            # Option pour modifier yolo.py si nécessaire
            if backup_dir and os.path.exists(os.path.join(backup_dir, 'yolo.py')):
                backup_yolo = os.path.join(backup_dir, 'yolo.py')
                print(f"  Restauration de {yolo_py_path} depuis {backup_yolo}...")
                shutil.copy(backup_yolo, yolo_py_path)
                print(f"  ✓ Restauré: {yolo_py_path}")
            else:
                print("  Tentative de modification automatique de yolo.py...")
                if 'from models.common import Conv, Bottleneck, SPP, DWConv, Focus, BottleneckCSP, C3' in yolo_content:
                    # Ajouter l'importation GD
                    yolo_content = yolo_content.replace(
                        'from models.common import Conv, Bottleneck, SPP, DWConv, Focus, BottleneckCSP, C3', 
                        'from models.common import Conv, Bottleneck, SPP, DWConv, Focus, BottleneckCSP, C3\nfrom models.gd import GDFusion, AttentionFusion, TransformerFusion'
                    )
                    
                    # Ajouter le support pour GD dans parse_model
                    if 'elif m is Detect:' in yolo_content:
                        yolo_content = yolo_content.replace(
                            'elif m is Detect:', 
                            'elif m is GDFusion:\n            c2 = args[0]  # nombre de canaux de sortie\n        elif m is Detect:'
                        )
                        
                        # Écrire les modifications
                        with open(yolo_py_path, 'w') as f:
                            f.write(yolo_content)
                        print("  ✓ Modifications appliquées à yolo.py")
                    else:
                        print("  ✗ Impossible de trouver le point d'insertion dans parse_model")
                else:
                    print("  ✗ Impossible de trouver le point d'insertion pour les imports")
    else:
        print(f"\n✗ Fichier {yolo_py_path} introuvable")
    
    print("\n=== Fin de la vérification ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vérifier les fichiers ADYOLOv5-Face")
    parser.add_argument('--yolo-dir', type=str, default='/content/yolov5-face',
                        help='Répertoire de YOLOv5-Face')
    parser.add_argument('--backup-dir', type=str, default='/content/drive/MyDrive/adyolov5_files',
                        help='Répertoire de sauvegarde des fichiers personnalisés')
    
    args = parser.parse_args()
    verify_adyolov5_files(args.yolo_dir, args.backup_dir)
