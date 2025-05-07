#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script tout-en-un pour Google Colab
Ce script télécharge et exécute tous les composants du projet YOLOv5-Face
"""

import os
import subprocess
import sys
import argparse
import time

def setup_environment(yolo_version='5.0', model_size='s', batch_size=32, img_size=640, epochs=50):
    """Configuration complète de l'environnement et lancement de l'entraînement"""
    
    start_time = time.time()
    
    # 1. Monter Google Drive
    try:
        from google.colab import drive
        print("=== Montage de Google Drive ===")
        drive.mount('/content/drive')
    except ImportError:
        print("⚠️ Ce script doit être exécuté dans Google Colab pour fonctionner correctement.")
        return False
    
    # 2. Installer numpy, scipy, etc. avec les versions compatibles
    print("\n=== Installation des dépendances compatibles ===")
    subprocess.run(['pip', 'install', 'numpy==1.26.4', 'scipy==1.13.1', 'gensim==4.3.3', '--no-deps'], check=True)
    
    # 3. Cloner le dépôt YOLOv5-Face
    yolo_dir = '/content/yolov5-face'
    if os.path.exists(yolo_dir):
        print(f"\n=== Suppression du dépôt existant ===")
        subprocess.run(['rm', '-rf', yolo_dir], check=True)
    
    print("\n=== Clonage du dépôt YOLOv5-Face ===")
    subprocess.run(['git', 'clone', 'https://github.com/deepcam-cn/yolov5-face.git', yolo_dir], check=True)
    
    # 4. Créer et configurer requirements.txt
    print("\n=== Création du fichier requirements.txt ===")
    requirements = """torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.5.0
tqdm>=4.64.0
PyYAML>=6.0
seaborn>=0.12.0
scipy<=1.13.1
thop>=0.1.1
requests>=2.27.0
Cython>=0.29.0
onnx>=1.12.0
onnxruntime>=1.10.0
tensorboard>=2.8.0
gensim==4.3.3"""
    
    with open(f'{yolo_dir}/requirements.txt', 'w') as f:
        f.write(requirements)
    
    # 5. Installer les dépendances
    print("\n=== Installation des dépendances ===")
    subprocess.run(['pip', 'install', '-r', f'{yolo_dir}/requirements.txt', '--no-deps'], check=True)
    subprocess.run(['pip', 'install', 'torch>=2.0.0', 'torchvision>=0.15.0'], check=True)
    
    # 6. Télécharger les poids pré-entraînés
    print(f"\n=== Téléchargement des poids YOLOv5 v{yolo_version} ===")
    weights_dir = os.path.join(yolo_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    
    weights_url = f'https://github.com/ultralytics/yolov5/releases/download/v{yolo_version}/yolov5{model_size}.pt'
    weights_path = os.path.join(weights_dir, f'yolov5{model_size}.pt')
    
    subprocess.run(['wget', weights_url, '-O', weights_path], check=True)
    
    # 7. Patcher le script train.py pour PyTorch 2.6+
    print("\n=== Modification du script train.py pour PyTorch 2.6+ ===")
    train_path = os.path.join(yolo_dir, 'train.py')
    
    if os.path.exists(train_path):
        with open(train_path, 'r') as f:
            content = f.read()
        
        modified_content = content.replace(
            'torch.load(weights, map_location=device)',
            'torch.load(weights, map_location=device, weights_only=False)'
        )
        
        with open(train_path, 'w') as f:
            f.write(modified_content)
        print(f"✓ Fichier {train_path} modifié avec succès")
    
    # 8. Corriger les erreurs de NumPy dans les fichiers Python
    print("\n=== Correction des erreurs de NumPy API ===")
    fixed_files = 0
    
    for root, dirs, files in os.walk(yolo_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    if "np.int" in content and not "np.int32" in content and not "np.int64" in content:
                        modified_content = content.replace("np.int", "np.int32")
                        
                        with open(file_path, 'w') as f:
                            f.write(modified_content)
                        
                        print(f"✓ Correction effectuée dans {os.path.relpath(file_path, yolo_dir)}")
                        fixed_files += 1
                except Exception:
                    pass
    
    if fixed_files > 0:
        print(f"✓ {fixed_files} fichiers corrigés avec succès")
    else:
        print("✓ Aucune correction nécessaire")
    
    # 9. Créer le fichier de configuration YAML pour WIDER Face
    print("\n=== Création du fichier de configuration YAML ===")
    data_dir = f'{yolo_dir}/data/widerface'
    
    yaml_content = {
        'train': f'{yolo_dir}/data/widerface/train',
        'val': f'{yolo_dir}/data/widerface/val',
        'test': f'{yolo_dir}/data/widerface/test',
        'nc': 1,
        'names': ['face'],
        'img_size': [img_size, img_size],
        'conf_thres': 0.25,
        'iou_thres': 0.45,
        'path': f'{yolo_dir}/data/widerface'
    }
    
    yaml_path = f'{yolo_dir}/data/widerface.yaml'
    os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
    
    import yaml
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    
    # 10. Préparation des données WIDER Face
    print("\n=== Préparation des données WIDER Face ===")
    print("Vous devez avoir les fichiers suivants dans votre Google Drive:")
    print("- /content/drive/MyDrive/dataset/WIDER_train.zip")
    print("- /content/drive/MyDrive/dataset/WIDER_val.zip")
    print("- /content/drive/MyDrive/dataset/WIDER_test.zip")
    print("- /content/drive/MyDrive/dataset/retinaface_gt.zip")
    
    # 11. Lancer l'entraînement avec tous les paramètres
    print("\n=== Lancement de l'entraînement ===")
    
    # Vérifier que les fichiers zip existent
    for zip_file in ['WIDER_train.zip', 'WIDER_val.zip', 'WIDER_test.zip', 'retinaface_gt.zip']:
        if not os.path.exists(f'/content/drive/MyDrive/dataset/{zip_file}'):
            print(f"⚠️ Fichier manquant: /content/drive/MyDrive/dataset/{zip_file}")
            print("Veuillez vous assurer que tous les fichiers nécessaires sont disponibles.")
            return False
    
    # Créer les répertoires nécessaires
    for path in [
        f'{data_dir}/tmp/train/images',
        f'{data_dir}/tmp/val/images',
        f'{data_dir}/tmp/test/images',
        f'{data_dir}/train/images',
        f'{data_dir}/train/labels',
        f'{data_dir}/val/images',
        f'{data_dir}/val/labels',
        f'{data_dir}/test/images',
        f'{data_dir}/test/labels'
    ]:
        os.makedirs(path, exist_ok=True)
    
    # Extraction des fichiers ZIP
    for zip_name, extract_path in [
        ('WIDER_train.zip', f'{data_dir}/tmp/'),
        ('WIDER_val.zip', f'{data_dir}/tmp/'),
        ('WIDER_test.zip', f'{data_dir}/tmp/'),
        ('retinaface_gt.zip', f'{data_dir}/tmp/')
    ]:
        zip_path = f'/content/drive/MyDrive/dataset/{zip_name}'
        print(f"Extraction de {zip_name}...")
        try:
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            print(f"✓ Extraction réussie: {zip_path}")
        except Exception as e:
            print(f"✗ Erreur lors de l'extraction de {zip_path}: {e}")
            return False
    
    # Copie des images vers la structure finale
    for src_dir, dst_dir in [
        (f'{data_dir}/tmp/WIDER_train/images', f'{data_dir}/train/images/'),
        (f'{data_dir}/tmp/WIDER_val/images', f'{data_dir}/val/images/'),
        (f'{data_dir}/tmp/WIDER_test/images', f'{data_dir}/test/images/')
    ]:
        if os.path.exists(src_dir):
            print(f"Copie de {src_dir} vers {dst_dir}")
            for root, dirs, files in os.walk(src_dir):
                rel_path = os.path.relpath(root, src_dir)
                if rel_path == '.':
                    rel_path = ''
                
                # Création des répertoires
                for d in dirs:
                    os.makedirs(os.path.join(dst_dir, rel_path, d), exist_ok=True)
                
                # Copie des fichiers
                for f in files:
                    src_file = os.path.join(root, f)
                    dst_file = os.path.join(dst_dir, rel_path, f)
                    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                    try:
                        import shutil
                        shutil.copy2(src_file, dst_file)
                    except Exception as e:
                        print(f"✗ Erreur lors de la copie de {src_file}: {e}")
            
            print(f"✓ Images copiées de {src_dir} vers {dst_dir}")
        else:
            print(f"✗ Répertoire source non trouvé: {src_dir}")
            return False
    
    # Conversion des annotations
    for set_name in ['train', 'val']:
        label_path = f'{data_dir}/tmp/{set_name}/label.txt'
        images_dir = f'{data_dir}/{set_name}/images'
        labels_dir = f'{data_dir}/{set_name}/labels'
        
        if os.path.exists(label_path):
            print(f"Traitement des annotations {set_name}...")
            try:
                # Code simplifié pour la conversion des annotations
                # Voir la fonction convert_annotations dans data_preparation.py pour l'implémentation complète
                import cv2
                
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                current_image = None
                current_labels = []
                processed_files = 0
                skipped_files = 0
                
                for line in lines:
                    line = line.strip()
                    
                    if line.startswith('#'):
                        # Si nous avons déjà une image en cours de traitement, sauvegardons ses annotations
                        if current_image is not None:
                            # Sauvegarder les annotations YOLO (code simplifié)
                            img_path = os.path.join(images_dir, current_image)
                            if os.path.exists(img_path):
                                img = cv2.imread(img_path)
                                if img is not None:
                                    height, width, _ = img.shape
                                    
                                    # Créer le fichier d'annotations YOLO
                                    base_name = os.path.splitext(current_image)[0]
                                    label_path = os.path.join(labels_dir, base_name + '.txt')
                                    
                                    # Créer le répertoire parent si nécessaire
                                    os.makedirs(os.path.dirname(label_path), exist_ok=True)
                                    
                                    valid_annotations = []
                                    
                                    for label in current_labels:
                                        # Format YOLO: class x_center y_center width height
                                        try:
                                            x = float(label[0])
                                            y = float(label[1])
                                            w = float(label[2])
                                            h = float(label[3])
                                            
                                            # Vérifier les dimensions
                                            if w <= 0 or h <= 0:
                                                continue
                                            
                                            # Conversion en format YOLO (normalisé)
                                            x_center = (x + w/2) / width
                                            y_center = (y + h/2) / height
                                            w_norm = w / width
                                            h_norm = h / height
                                            
                                            # Vérifier que les coordonnées normalisées sont valides
                                            if (x_center < 0 or x_center > 1 or 
                                                y_center < 0 or y_center > 1 or 
                                                w_norm <= 0 or w_norm > 1 or 
                                                h_norm <= 0 or h_norm > 1):
                                                continue
                                            
                                            # Ajouter les points de repère (landmarks) si disponibles
                                            landmarks = []
                                            if len(label) >= 19:  # Si nous avons des points de repère
                                                for i in range(5):
                                                    lm_x = float(label[4 + i*3])
                                                    lm_y = float(label[5 + i*3])
                                                    visible = float(label[6 + i*3])  # 1=visible, 0=invisible, -1=absent
                                                    
                                                    # Normalisation
                                                    lm_x_norm = lm_x / width
                                                    lm_y_norm = lm_y / height
                                                    
                                                    # Si point non visible, utiliser -1
                                                    if visible != 1:
                                                        lm_x_norm = -1
                                                        lm_y_norm = -1
                                                    
                                                    landmarks.extend([lm_x_norm, lm_y_norm])
                                            else:
                                                # Si aucun point de repère, utiliser -1 pour tous
                                                landmarks = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
                                            
                                            # Format de l'annotation YOLO avec landmarks
                                            yolo_format = (
                                                f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f} "
                                                f"{landmarks[0]:.6f} {landmarks[1]:.6f} {landmarks[2]:.6f} {landmarks[3]:.6f} "
                                                f"{landmarks[4]:.6f} {landmarks[5]:.6f} {landmarks[6]:.6f} {landmarks[7]:.6f} "
                                                f"{landmarks[8]:.6f} {landmarks[9]:.6f}"
                                            )
                                            
                                            valid_annotations.append(yolo_format)
                                        except Exception:
                                            # Ignorer les annotations mal formatées
                                            continue
                                    
                                    # Écrire les annotations valides
                                    with open(label_path, 'w') as f:
                                        if valid_annotations:
                                            f.write('\n'.join(valid_annotations))
                                            processed_files += 1
                                        else:
                                            skipped_files += 1
                        
                        # Nouvelle image
                        image_path = line[2:]
                        current_image = image_path
                        current_labels = []
                    else:
                        # Ajout d'une annotation
                        values = line.split()
                        if len(values) >= 4:  # Au moins x, y, w, h
                            current_labels.append(values)
                
                print(f"✓ Conversion terminée: {processed_files} fichiers traités, {skipped_files} fichiers ignorés")
            except Exception as e:
                print(f"✗ Erreur lors de la conversion des annotations {set_name}: {e}")
                return False
        else:
            print(f"✗ Fichier d'annotations non trouvé: {label_path}")
            return False
    
    # Filtrage des images corrompues
    print("\n=== Filtrage des images corrompues ===")
    for set_name in ['train', 'val']:
        dataset_path = f'{data_dir}/{set_name}'
        labels_dir = f'{dataset_path}/labels'
        images_dir = f'{dataset_path}/images'
        
        if not os.path.exists(labels_dir):
            print(f"✗ Répertoire de labels non trouvé: {labels_dir}")
            continue
        
        corrupted_count = 0
        fixed_count = 0
        
        # Parcourir tous les fichiers d'étiquettes
        for root, _, files in os.walk(labels_dir):
            for file in files:
                if file.endswith('.txt'):
                    label_path = os.path.join(root, file)
                    rel_path = os.path.relpath(label_path, labels_dir)
                    
                    # Vérifier l'image correspondante
                    image_name = os.path.splitext(rel_path)[0] + '.jpg'
                    image_path = os.path.join(images_dir, image_name)
                    
                    if not os.path.exists(image_path):
                        corrupted_count += 1
                        continue
                    
                    # Vérifier et corriger les annotations
                    try:
                        with open(label_path, 'r') as f:
                            lines = f.readlines()
                        
                        # Vérifier chaque ligne d'annotation
                        valid_lines = []
                        has_invalid = False
                        
                        for line in lines:
                            parts = line.strip().split()
                            
                            if len(parts) < 15:  # Format YOLOv5-face complet (1 classe + 4 bbox + 10 landmarks)
                                has_invalid = True
                                continue
                            
                            # Vérifier les coordonnées normalisées
                            try:
                                class_id = int(parts[0])
                                x, y, w, h = map(float, parts[1:5])
                                
                                # Valider les coordonnées
                                if (x < 0 or x > 1 or y < 0 or y > 1 or 
                                    w <= 0 or w > 1 or h <= 0 or h > 1):
                                    has_invalid = True
                                    continue
                                
                                valid_lines.append(line)
                            except ValueError:
                                has_invalid = True
                                continue
                        
                        # Sauvegarder le fichier corrigé si nécessaire
                        if has_invalid:
                            with open(label_path, 'w') as f:
                                f.writelines(valid_lines)
                            fixed_count += 1
                    
                    except Exception:
                        corrupted_count += 1
        
        print(f"✓ Filtrage terminé pour {set_name}: {corrupted_count} fichiers corrompus, {fixed_count} fichiers corrigés.")
    
    # Lancement de l'entraînement
    print("\n=== Début de l'entraînement ===")
    train_cmd = [
        'python', f'{yolo_dir}/train.py',
        '--data', yaml_path,
        '--cfg', f'{yolo_dir}/models/yolov5{model_size}.yaml',
        '--weights', weights_path,
        '--batch-size', str(batch_size),
        '--epochs', str(epochs),
        '--img', str(img_size),
        '--project', f'{yolo_dir}/runs/train',
        '--name', 'face_detection_transfer',
        '--exist-ok',
        '--cache',
        '--rect'
    ]
    
    # Afficher la commande
    print("Commande d'entraînement:")
    print(' '.join(train_cmd))
    
    # Lancer l'entraînement
    try:
        subprocess.run(train_cmd, check=True)
        print("\n✓ Entraînement terminé avec succès!")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Erreur lors de l'entraînement: {e}")
        return False
    
    # Afficher le temps total d'exécution
    end_time = time.time()
    duration = end_time - start_time
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print("\n" + "=" * 80)
    print(f"PIPELINE TERMINÉ EN {int(hours)}h {int(minutes)}m {int(seconds)}s")
    print("=" * 80)
    
    print("\nRésultats et modèles disponibles dans:")
    print(f"- Modèle PyTorch: {yolo_dir}/runs/train/face_detection_transfer/weights/best.pt")
    print(f"- Métriques et logs: {yolo_dir}/runs/train/face_detection_transfer")
    print(f"\nPour visualiser avec TensorBoard, exécutez:")
    print(f"%load_ext tensorboard")
    print(f"%tensorboard --logdir {yolo_dir}/runs/train/face_detection_transfer")
    
    # Proposer d'exporter le modèle au format ONNX
    print("\nPour exporter le modèle au format ONNX, exécutez:")
    print(f"!python {yolo_dir}/export.py --weights {yolo_dir}/runs/train/face_detection_transfer/weights/best.pt --img {img_size} --batch-size 1 --dynamic")
    
    return True

def parse_args():
    """Analyse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(description='Configuration Colab pour YOLOv5-Face')
    
    parser.add_argument('--yolo-version', type=str, default='5.0',
                        help='Version de YOLOv5 à utiliser (par exemple 5.0)')
    parser.add_argument('--model-size', type=str, default='s', choices=['s', 'm', 'l', 'x'],
                        help='Taille du modèle YOLOv5 (s, m, l, x)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Taille du batch pour l\'entraînement')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Nombre d\'epochs d\'entraînement')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Taille d\'image pour l\'entraînement')
    
    return parser.parse_args()

if __name__ == "__main__":
    # Analyse des arguments
    args = parse_args()
    
    # Configuration de l'environnement et lancement de l'entraînement
    setup_environment(
        yolo_version=args.yolo_version,
        model_size=args.model_size,
        batch_size=args.batch_size,
        img_size=args.img_size,
        epochs=args.epochs
    )
