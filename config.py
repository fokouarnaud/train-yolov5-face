#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Configuration centralisée pour le projet YOLOv5-Face
Contient les paramètres globaux pour faciliter la maintenance
"""

# URL du dépôt GitHub à utiliser (forké avec les corrections)
REPO_URL = "https://github.com/fokouarnaud/yolov5-face.git"

# Versions des dépendances
DEPENDENCIES = {
    "numpy": "1.26.4",
    "scipy": "1.13.1",
    "gensim": "4.3.3",
    "torch": "2.0.0",
    "torchvision": "0.15.0",
    "opencv-python": "4.5.0",
}

# Répertoires par défaut
DEFAULT_PATHS = {
    "root_dir": "/content",
    "yolo_dir": "/content/yolov5-face",
    "weights_dir": "/content/yolov5-face/weights",
    "data_dir": "/content/yolov5-face/data/widerface",
    "drive_dataset_path": "/content/drive/MyDrive/dataset",
    "results_dir": "/content/drive/MyDrive/YOLOv5_Face_Results",
}

# Configuration de l'entraînement par défaut
DEFAULT_TRAINING = {
    "batch_size": 40,
    "epochs": 300,
    "img_size": 640,
    "model_size": "s",
    "yolo_version": "5.0",
}

# Messages d'information
INFO_MESSAGES = {
    "pytorch_fix": "✓ Le dépôt forké inclut déjà les corrections nécessaires pour PyTorch 2.6+",
    "numpy_fix": "✓ Correction des problèmes liés à np.int dans NumPy 1.26+",
    "adyolo_info": "ADYOLOv5-Face est une version améliorée avec un mécanisme Gather-and-Distribute pour la détection des petits visages",
}

# Configuration des scripts à copier
REQUIRED_SCRIPTS = [
    "main.py",
    "data_preparation.py",
    "model_training.py",
    "model_evaluation.py",
    "utils.py",
    "colab_setup.py",
    "config.py",  # Inclure la config elle-même
]

# Modèles disponibles
AVAILABLE_MODELS = {
    "n-0.5": {
        "description": "Version ultra-légère avec ShuffleNetV2 (spécifique à YOLOv5-Face)",
        "use_case": "Appareils mobiles avec ressources limitées",
    },
    "n": {
        "description": "Version légère avec ShuffleNetV2 (spécifique à YOLOv5-Face)",
        "use_case": "Appareils mobiles et embarqués",
    },
    "s": {
        "description": "Version petite avec CSPNet",
        "use_case": "Équilibre entre vitesse et précision",
    },
    "m": {
        "description": "Version moyenne avec CSPNet",
        "use_case": "Précision accrue avec vitesse acceptable",
    },
    "l": {
        "description": "Version large avec CSPNet",
        "use_case": "Haute précision pour les cas d'utilisation exigeants",
    },
    "x": {
        "description": "Version extra-large avec CSPNet",
        "use_case": "Précision maximale, vitesse réduite",
    },
    "ad": {
        "description": "ADYOLOv5-Face avec mécanisme Gather-and-Distribute",
        "use_case": "Détection optimisée des petits visages",
    },
}
