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

# Configuration de l'entraînement par défaut (conforme à l'article ADYOLOv5-Face)
DEFAULT_TRAINING = {
    "batch_size": 32,  # Conforme à l'article (32) - OPTIMISÉ
    "epochs": 250,     # Conforme à l'article (250) - OPTIMISÉ
    "img_size": 640,   # Conforme à l'article (640) - OPTIMISÉ
    "model_size": "s",
    "yolo_version": "5.0",
}

# Configuration alternative pour environnements avec contraintes mémoire
MEMORY_OPTIMIZED_TRAINING = {
    "batch_size": 24,  # OPTIMISÉ: équilibre entre performance et mémoire
    "epochs": 150,     # OPTIMISÉ: compromis performance/temps
    "img_size": 576,   # OPTIMISÉ: meilleur compromis pour petits visages
    "model_size": "s",
    "yolo_version": "5.0",
}

# Configuration des modèles disponibles
MODEL_CONFIGS = {
    "n-0.5": {
        "yaml": "yolov5n-0.5.yaml",
        "weights": "yolov5n-0.5.pt",
        "img_size": 640,
    },
    "n": {
        "yaml": "yolov5n.yaml",
        "weights": "yolov5n.pt",
        "img_size": 640,
    },
    "s": {
        "yaml": "yolov5s.yaml",
        "weights": "yolov5s.pt",
        "img_size": 640,
    },
    "s6": {
        "yaml": "yolov5s6.yaml",
        "weights": "yolov5s6.pt",
        "img_size": 640,
    },
    "m": {
        "yaml": "yolov5m.yaml",
        "weights": "yolov5m.pt",
        "img_size": 640,
    },
    "m6": {
        "yaml": "yolov5m6.yaml",
        "weights": "yolov5m6.pt",
        "img_size": 640,
    },
    "l": {
        "yaml": "yolov5l.yaml",
        "weights": "yolov5l.pt",
        "img_size": 640,
    },
    "l6": {
        "yaml": "yolov5l6.yaml",
        "weights": "yolov5l6.pt",
        "img_size": 640,
    },
    "ad": {  # ADYOLOv5-Face (conforme à l'article)
        "yaml": "adyolov5s.yaml",
        "weights": "yolov5s.pt",  # Utilise les poids de base de YOLOv5s
        "img_size": 640,  # Conforme à l'article ADYOLOv5-Face
    },
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
