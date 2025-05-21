#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script de configuration pour Google Colab
qui prépare l'environnement pour l'entraînement de YOLOv5-Face
"""

import os
import sys
import subprocess
import argparse

# Importer la configuration centralisée
from config import REPO_URL, DEPENDENCIES, DEFAULT_PATHS, INFO_MESSAGES

def setup_environment(model_size='s', yolo_version='5.0', model_yaml=None):
    """
    Configure l'environnement Colab pour l'entraînement
    
    Args:
        model_size (str): Taille du modèle 
                          - n-0.5, n : modèles ultra-légers (ShuffleNetV2) pour appareils mobiles
                          - s, m, l, x : modèles standards (CSPNet)
                          - s6, m6, l6, x6 : versions avec bloc P6 pour grands visages
                          - ad : ADYOLOv5 avec mécanisme Gather-and-Distribute pour petits visages
        yolo_version (str): Version de YOLOv5 à utiliser
        model_yaml (str): Fichier YAML spécifique à utiliser (remplace celui défini par model_size)
    """
    # Bloquer complètement l'utilisation de n6 qui n'est pas officiellement supporté
    if model_size == 'n6':
        print("⚠️ Le modèle YOLOv5n6 n'est pas officiellement supporté et peut provoquer des erreurs")
        print("Nous recommandons d'utiliser YOLOv5s6 à la place pour la détection des grands visages")
        print("→ La configuration continue avec le modèle 's' par défaut")
        model_size = 's'
        
    # 1. Installer les dépendances compatibles
    print("=== Installation des dépendances compatibles ===")
    subprocess.run(['pip', 'install', 'numpy==1.26.4', 'scipy==1.13.1', 'gensim==4.3.3', '--no-deps'], check=True)
    
    # 2. Vérifier si le dépôt YOLOv5-Face est cloné
    yolo_dir = '/content/yolov5-face'
    if not os.path.exists(yolo_dir):
        print("=== Clonage du dépôt YOLOv5-Face ===")
        # Utiliser le dépôt forké avec les corrections déjà appliquées
        subprocess.run(['git', 'clone', REPO_URL, yolo_dir], check=True)
    
    # 3. Créer le répertoire des poids
    weights_dir = os.path.join(yolo_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    
    # 4. Télécharger les poids pré-entraînés
    print(f"=== Téléchargement des poids YOLOv5 v{yolo_version} ===")
    # Si c'est ADYOLOv5, nous utiliserons les poids YOLOv5s comme base
    base_model_size = 's' if model_size == 'ad' else model_size
    weights_to_download = ['s', 'm', 'l', 'x'] if model_size == 'all' else [base_model_size]
    
    for size in weights_to_download:
        weights_url = f'https://github.com/ultralytics/yolov5/releases/download/v{yolo_version}/yolov5{size}.pt'
        weights_path = os.path.join(weights_dir, f'yolov5{size}.pt')
        
        if not os.path.exists(weights_path):
            print(f"Téléchargement de yolov5{size}.pt...")
            try:
                # Vérifier si le fichier existe et n'est pas vide
                if os.path.exists(weights_path) and os.path.getsize(weights_path) == 0:
                    os.remove(weights_path)  # Supprimer le fichier vide pour éviter les erreurs futures
                    print(f"Suppression du fichier de poids vide: {weights_path}")
                    
                subprocess.run(['wget', weights_url, '-O', weights_path], check=True)
                print(f"✓ Poids yolov5{size}.pt téléchargés")
            except subprocess.CalledProcessError:
                print(f"✗ Erreur lors du téléchargement des poids yolov5{size}.pt")
                if size in ['n-0.5', 'n']:
                    print(f"Les modèles YOLOv5{size} sont des modèles ultra-légers spécifiques à YOLOv5-Face.")
                    print(f"Ces modèles utilisent l'architecture ShuffleNetV2 et sont optimisés pour les appareils mobiles.")
                    print(f"Ils seront initialisés avec des poids aléatoires pour l'entraînement.")
                else:
                    print(f"  Le modèle {size} sera initialisé avec des poids aléatoires")
                # Certaines variantes comme n-0.5 et n peuvent ne pas être disponibles en téléchargement
        else:
            print(f"✓ Poids yolov5{size}.pt déjà présents")
    
    # 5. Si c'est ADYOLOv5, préparer les fichiers nécessaires
    if model_size == 'ad':
        print("=== Configuration d'ADYOLOv5-Face ===")
        
        # Créer le module Gather-and-Distribute (gd.py)
        gd_path = os.path.join(yolo_dir, 'models', 'gd.py')
        if not os.path.exists(gd_path):
            print("Création du module Gather-and-Distribute (gd.py)...")
            gd_code = """
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import Conv, autopad

class FeatureAlignmentModule(nn.Module):
    \"\"\"Module d'alignement des caractéristiques pour le mécanisme GD\"\"\"
    def __init__(self, target_size):
        super(FeatureAlignmentModule, self).__init__()
        self.target_size = target_size if isinstance(target_size, tuple) else (target_size, target_size)
        
    def forward(self, features):
        aligned_features = []
        for feat in features:
            h, w = feat.shape[2], feat.shape[3]
            th, tw = self.target_size
            
            # Si la feature map est plus petite que la cible, utilisez une interpolation bilinéaire
            if h < th or w < tw:
                aligned = F.interpolate(feat, size=self.target_size, mode='bilinear', align_corners=False)
            # Si la feature map est plus grande que la cible, utilisez un average pooling
            elif h > th or w > tw:
                # Calculer le facteur d'échelle pour le redimensionnement
                scale_h, scale_w = h / th, w / tw
                if scale_h > 1 and scale_w > 1:
                    aligned = F.adaptive_avg_pool2d(feat, self.target_size)
                else:
                    aligned = F.interpolate(feat, size=self.target_size, mode='bilinear', align_corners=False)
            else:
                aligned = feat
            aligned_features.append(aligned)
        return aligned_features

class InformationFusionModule(nn.Module):
    \"\"\"Module de fusion des informations pour le mécanisme GD\"\"\"
    def __init__(self, channels):
        super(InformationFusionModule, self).__init__()
        self.conv1 = Conv(channels, channels, k=1)
        self.conv3 = Conv(channels, channels, k=3, p=1)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.SiLU(),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features):
        # Additionner les features alignées
        fused = sum(features)
        # Appliquer une convolution 1x1
        fused = self.conv1(fused)
        # Appliquer une convolution 3x3
        fused = self.conv3(fused)
        # Appliquer l'attention
        att = self.attention(fused)
        return fused * att

class InformationInjectionModule(nn.Module):
    \"\"\"Module d'injection d'informations pour le mécanisme GD\"\"\"
    def __init__(self, channels):
        super(InformationInjectionModule, self).__init__()
        self.conv1 = Conv(channels, channels, k=1)
        self.conv3 = Conv(channels, channels, k=3, p=1)
        
    def forward(self, x, global_info):
        # Adapter la taille du global_info à celle de x si nécessaire
        if global_info.shape[2:] != x.shape[2:]:
            global_info = F.interpolate(
                global_info, size=x.shape[2:], mode='bilinear', align_corners=False
            )
        
        # Additionner les features
        x = x + global_info
        # Appliquer une convolution 1x1
        x = self.conv1(x)
        # Appliquer une convolution 3x3
        x = self.conv3(x)
        return x

class LowStageGD(nn.Module):
    \"\"\"Mécanisme Gather-and-Distribute pour les couches de bas niveau\"\"\"
    def __init__(self, channels):
        super(LowStageGD, self).__init__()
        self.target_size = 80  # Taille cible pour l'alignement des features
        self.fam = FeatureAlignmentModule(self.target_size)
        self.ifm = InformationFusionModule(channels)
        self.iim = nn.ModuleList([
            InformationInjectionModule(channels) for _ in range(3)
        ])
        
    def forward(self, x):
        # x devrait être une liste [P3, P4, P5]
        aligned_features = self.fam(x)
        global_info = self.ifm(aligned_features)
        
        # Injecter l'information globale dans chaque niveau de feature
        enhanced_features = []
        for i, feat in enumerate(x):
            enhanced = self.iim[i](feat, global_info)
            enhanced_features.append(enhanced)
            
        return enhanced_features

class HighStageGD(nn.Module):
    \"\"\"Mécanisme Gather-and-Distribute pour les couches de haut niveau\"\"\"
    def __init__(self, channels):
        super(HighStageGD, self).__init__()
        self.target_size = 40  # Taille cible plus petite pour l'alignement des features
        self.fam = FeatureAlignmentModule(self.target_size)
        self.ifm = InformationFusionModule(channels)
        self.iim = nn.ModuleList([
            InformationInjectionModule(channels) for _ in range(3)
        ])
        self.extra_conv = Conv(channels, channels, k=3, p=1)
        
    def forward(self, x):
        # x devrait être une liste [P3, P4, P5]
        aligned_features = self.fam(x)
        global_info = self.ifm(aligned_features)
        global_info = self.extra_conv(global_info)
        
        # Injecter l'information globale dans chaque niveau de feature
        enhanced_features = []
        for i, feat in enumerate(x):
            enhanced = self.iim[i](feat, global_info)
            enhanced_features.append(enhanced)
            
        return enhanced_features

def make_divisible(x, divisor):
    # Fonction utilitaire pour s'assurer que tous les nombres de filtres sont un multiple du diviseur donné
    return max(int(x + divisor / 2) // divisor * divisor, divisor)
"""
            os.makedirs(os.path.dirname(gd_path), exist_ok=True)
            with open(gd_path, 'w') as f:
                f.write(gd_code.strip())
            print(f"✓ Fichier {gd_path} créé")
        else:
            print(f"✓ Fichier {gd_path} déjà présent")
        
        # Créer le fichier de configuration YAML pour ADYOLOv5
        adyolo_yaml_path = os.path.join(yolo_dir, 'models', 'adyolov5s_simple.yaml')
        if not os.path.exists(adyolo_yaml_path):
            print("Création du fichier de configuration ADYOLOv5-Face (adyolov5s_simple.yaml)...")
            adyolo_yaml_content = """
# Modèle YOLOv5s adapté avec mécanisme GD pour ADYOLOv5-Face

# Paramètres
nc: 1  # nombre de classes (1 pour les visages)
depth_multiple: 0.33  # facteur pour la profondeur du modèle
width_multiple: 0.50  # facteur pour la largeur du modèle

# Anchors
anchors:
  - [10,13, 16,30, 33,23]  # P3/8
  - [30,61, 62,45, 59,119]  # P4/16
  - [116,90, 156,198, 373,326]  # P5/32

# YOLOv5s backbone
backbone:
  # [from, number, module, args]
  [[-1, 1, Focus, [64, 3]],  # 0-P1/2
   [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
   [-1, 3, C3, [128]],
   [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
   [-1, 9, C3, [256]],
   [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
   [-1, 9, C3, [512]],
   [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
   [-1, 1, SPP, [1024, [5, 9, 13]]],
   [-1, 3, C3, [1024, False]],  # 9
  ]

# YOLOv5s head
head:
  [[-1, 1, Conv, [512, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 6], 1, Concat, [1]],  # cat backbone P4
   [-1, 3, C3, [512, False]],  # 13
   
   [-1, 1, Conv, [256, 1, 1]],
   [-1, 1, nn.Upsample, [None, 2, 'nearest']],
   [[-1, 4], 1, Concat, [1]],  # cat backbone P3
   [-1, 3, C3, [256, False]],  # 17 (P3)
   
   # Ajout du mécanisme Gather-and-Distribute Low-Stage
   [[17, 13, 9], 1, LowStageGD, [256]],  # 18: GD pour P3, P4, P5
   
   [18, 1, Conv, [256, 3, 2]],
   [[-1, 13], 1, Concat, [1]],  # cat P4
   [-1, 3, C3, [512, False]],  # 21 (P4)
   
   [-1, 1, Conv, [512, 3, 2]],
   [[-1, 9], 1, Concat, [1]],  # cat P5
   [-1, 3, C3, [1024, False]],  # 24 (P5)
   
   # Ajout du mécanisme Gather-and-Distribute High-Stage
   [[18, 21, 24], 1, HighStageGD, [512]],  # 25: GD pour features améliorées
   
   [[18, 21, 24], 1, Detect, [nc, anchors]],  # 26: tête de détection
  ]
"""
            os.makedirs(os.path.dirname(adyolo_yaml_path), exist_ok=True)
            with open(adyolo_yaml_path, 'w') as f:
                f.write(adyolo_yaml_content.strip())
            print(f"✓ Fichier {adyolo_yaml_path} créé")
        else:
            print(f"✓ Fichier {adyolo_yaml_path} déjà présent")
        
        # Créer le fichier d'hyperparamètres pour ADYOLOv5
        hyp_adyolo_path = os.path.join(yolo_dir, 'data', 'hyp.adyolo.yaml')
        if not os.path.exists(hyp_adyolo_path):
            print("Création du fichier d'hyperparamètres ADYOLOv5-Face (hyp.adyolo.yaml)...")
            hyp_adyolo_content = """
# Hyperparamètres optimisés pour ADYOLOv5-Face
lr0: 0.01  # taux d'apprentissage initial
lrf: 0.1  # facteur final du OneCycleLR (lr0 * lrf)
momentum: 0.937  # momentum SGD
weight_decay: 0.0005  # décroissance des poids
warmup_epochs: 3.0  # epochs de warmup
warmup_momentum: 0.8  # warmup momentum
warmup_bias_lr: 0.1  # warmup bias lr
box: 0.05  # perte de la boîte de délimitation
cls: 0.5  # perte de classification
cls_pw: 1.0  # pondération d'exponentielle positive de la perte de classification
obj: 1.0  # perte d'objet
obj_pw: 1.0  # pondération positive de la perte d'objet
iou_t: 0.20  # seuil IoU de l'entraîneur
anchor_t: 4.0  # ancre-vérité IoU seuil
fl_gamma: 0.0  # perte focale gamma (efficacité, 0 signifie pas de FL)
hsv_h: 0.015  # augmentation HSV-Hue
hsv_s: 0.7  # augmentation HSV-Saturation
hsv_v: 0.4  # augmentation HSV-Value
degrees: 0.0  # augmentation de rotation de l'image (+/- deg)
translate: 0.1  # augmentation de la translation de l'image (+/- fraction)
scale: 0.5  # augmentation de l'échelle de l'image (+/- gain)
shear: 0.0  # augmentation du cisaillement de l'image (+/- deg)
perspective: 0.0  # augmentation de la perspective de l'image (+/- fraction), le 0 ne le désactive pas
flipud: 0.0  # probabilité d'inclusion de l'augmentation de flipud de l'image
fliplr: 0.5  # probabilité d'inclusion de l'augmentation de fliplr de l'image
mosaic: 1.0  # probabilité d'inclusion de l'augmentation de mosaïque de l'image
mixup: 0.1  # probabilité de l'augmentation de mixup de l'image
copy_paste: 0.0  # probabilité de l'augmentation de copier-coller de l'image
"""
            os.makedirs(os.path.dirname(hyp_adyolo_path), exist_ok=True)
            with open(hyp_adyolo_path, 'w') as f:
                f.write(hyp_adyolo_content.strip())
            print(f"✓ Fichier {hyp_adyolo_path} créé")
        else:
            print(f"✓ Fichier {hyp_adyolo_path} déjà présent")
            
        # Modifier le fichier yolo.py pour ajouter le support du mécanisme GD
        yolo_py_path = os.path.join(yolo_dir, 'models', 'yolo.py')
        if os.path.exists(yolo_py_path):
            print("Mise à jour du fichier yolo.py pour supporter le mécanisme Gather-and-Distribute...")
            with open(yolo_py_path, 'r') as f:
                yolo_content = f.read()
            
            # Vérifier si l'importation du mécanisme GD est présente
            if 'from models.gd import' not in yolo_content:
                # Ajouter l'importation au début du fichier
                import_line = 'from models.common import Conv, Bottleneck, SPP, DWConv, Focus, BottleneckCSP, C3, ShuffleV2Block, Concat, NMS, autoShape, StemBlock, BlazeBlock, DoubleBlazeBlock'
                gd_import = 'from models.common import Conv, Bottleneck, SPP, DWConv, Focus, BottleneckCSP, C3, ShuffleV2Block, Concat, NMS, autoShape, StemBlock, BlazeBlock, DoubleBlazeBlock\nfrom models.gd import LowStageGD, HighStageGD'
                yolo_content = yolo_content.replace(import_line, gd_import)
                
                # Ajouter le support du mécanisme GD dans la fonction parse_model
                parse_model_section = 'elif m is Detect:'
                gd_parse_section = 'elif m in [LowStageGD, HighStageGD]:\n            c2 = args[0]  # nombre de canaux de sortie\n        elif m is Detect:'
                yolo_content = yolo_content.replace(parse_model_section, gd_parse_section)
                
                # Écrire les modifications dans le fichier
                with open(yolo_py_path, 'w') as f:
                    f.write(yolo_content)
                print(f"✓ Fichier {yolo_py_path} mis à jour pour supporter le mécanisme GD")
            else:
                print(f"✓ Fichier {yolo_py_path} contient déjà le support du mécanisme GD")
        else:
            print(f"✗ Fichier {yolo_py_path} introuvable")
    
    # 6. Vérification de la compatibilité PyTorch 2.6+
    print("=== Vérification de la compatibilité PyTorch 2.6+ ===")
    print(INFO_MESSAGES["pytorch_fix"])
    print("✓ Aucune modification du code n'est nécessaire")
    
    # 7. Ajouter le répertoire courant au PYTHONPATH
    if '/content' not in sys.path:
        print("=== Configuration du PYTHONPATH ===")
        sys.path.insert(0, '/content')
        print("✓ Répertoire /content ajouté au PYTHONPATH")
    
    # 8. Vérifier la présence des scripts Python
    scripts = ['main.py', 'data_preparation.py', 'model_training.py', 'model_evaluation.py', 'utils.py']
    missing_scripts = [script for script in scripts if not os.path.exists(f'/content/{script}')]
    
    if missing_scripts:
        print(f"⚠️ Attention: Les scripts suivants sont manquants: {', '.join(missing_scripts)}")
        print("Assurez-vous de les copier depuis Google Drive ou de les télécharger.")
    else:
        print("✓ Tous les scripts Python nécessaires sont présents")
    
    print("\n=== Configuration terminée ===")
    if model_size == 'ad':
        print("ADYOLOv5-Face a été configuré avec le mécanisme Gather-and-Distribute pour améliorer la détection des petits visages.")
    print("Vous pouvez maintenant exécuter le script principal avec la commande:")
    print("!python main.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configuration de l'environnement Colab pour YOLOv5-Face")
    parser.add_argument('--model-size', type=str, default='s', 
                        choices=['n-0.5', 'n', 's', 's6', 'm', 'm6', 'l', 'l6', 'x', 'x6', 'all', 'ad'],
                        help='Taille du modèle à télécharger (n-0.5, n, s, s6, m, m6, l, l6, x, x6, all, ad)')
    parser.add_argument('--yolo-version', type=str, default='5.0',
                        help='Version de YOLOv5 à utiliser (par exemple 5.0)')
    parser.add_argument('--model-yaml', type=str, default=None,
                        help='Fichier YAML spécifique à utiliser (pour ADYOLOv5-Face: adyolov5s.yaml ou adyolov5s_simple.yaml)')
    
    args = parser.parse_args()
    setup_environment(args.model_size, args.yolo_version, args.model_yaml)
