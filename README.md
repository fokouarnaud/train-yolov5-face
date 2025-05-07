# YOLOv5-Face Trainer pour Google Colab

Ce projet est une implémentation structurée et corrigée pour l'entraînement du modèle YOLOv5-Face sur Google Colab, spécialisé dans la détection de visages en utilisant le dataset WIDER Face. Il inclut des correctifs importants pour résoudre les problèmes de compatibilité avec les versions récentes de PyTorch (2.6+).

## Structure du projet

Le projet est organisé en plusieurs modules pour faciliter la maintenance et la compréhension :

- `main.py` : Script principal pour exécuter l'ensemble du pipeline
- `data_preparation.py` : Gestion de la préparation des données WIDER Face
- `model_training.py` : Entraînement du modèle YOLOv5-Face
- `model_evaluation.py` : Évaluation et exportation du modèle
- `utils.py` : Fonctions utilitaires communes

## Prérequis

- Google Colab avec GPU activé
- Accès à Google Drive avec les fichiers du dataset WIDER Face :
  - `/content/drive/MyDrive/dataset/WIDER_train.zip`
  - `/content/drive/MyDrive/dataset/WIDER_val.zip`
  - `/content/drive/MyDrive/dataset/WIDER_test.zip`
  - `/content/drive/MyDrive/dataset/retinaface_gt.zip`

## Installation

1. Téléchargez tous les fichiers du projet sur votre ordinateur
2. Créez un nouveau notebook Google Colab
3. Uploadez tous les fichiers du projet dans le notebook via le menu "Fichiers"
4. Assurez-vous que tous les fichiers sont bien visibles dans l'environnement Colab

## Utilisation

### Méthode simple

Exécutez la cellule suivante dans votre notebook Colab pour lancer l'ensemble du pipeline :

```python
!python main.py
```

### Options avancées

Vous pouvez également personnaliser l'exécution avec différentes options :

```python
# Exemples d'utilisation avec des options personnalisées
!python main.py --batch-size 16 --epochs 100 --img-size 800 --model-size m
```

Options disponibles :
- `--batch-size` : Taille du batch pour l'entraînement (défaut: 32)
- `--epochs` : Nombre d'epochs d'entraînement (défaut: 300)
- `--img-size` : Taille d'image pour l'entraînement (défaut: 640)
- `--model-size` : Taille du modèle YOLOv5 (s, m, l, x) (défaut: s)
- `--skip-train` : Ignorer l'étape d'entraînement
- `--skip-evaluation` : Ignorer l'étape d'évaluation
- `--skip-export` : Ignorer l'étape d'exportation

## Corrections apportées

Cette version corrige plusieurs problèmes du script original :

1. **Problème API NumPy** : Correction de l'erreur liée à `np.int` déprécié (remplacé par `np.int32`)
2. **Gestion des images corrompues** : Amélioration du filtrage des images et annotations non valides
3. **Structure du code** : Organisation en modules pour une meilleure maintenabilité
4. **Robustesse des commandes** : Utilisation de `subprocess.run()` au lieu de commandes shell directes
5. **Gestion des erreurs** : Meilleure gestion des exceptions et messages d'erreur détaillés

## Flux de travail

1. **Configuration de l'environnement** : Installation des dépendances et préparation de l'environnement Colab
2. **Préparation des données** : Extraction et conversion des annotations WIDER Face au format YOLO
3. **Entraînement** : Transfert learning à partir des poids YOLOv5 pré-entraînés
4. **Évaluation** : Évaluation du modèle sur l'ensemble de validation WIDER Face
5. **Exportation** : Exportation du modèle au format ONNX pour le déploiement

## Résultats

Après l'exécution complète, vous trouverez les résultats dans les répertoires suivants :

- Modèle PyTorch : `/content/yolov5-face/runs/train/face_detection_transfer/weights/best.pt`
- Modèle ONNX : `/content/yolov5-face/runs/train/face_detection_transfer/weights/best.onnx`
- Métriques et logs : `/content/yolov5-face/runs/train/face_detection_transfer`
- Visualisation TensorBoard : Accessible via TensorBoard dans Colab

## Auteur

Ce code a été restructuré et corrigé pour une meilleure utilisation sur Google Colab.

## Licence

Ce projet est basé sur [YOLOv5-Face](https://github.com/deepcam-cn/yolov5-face) et suit la même licence que le projet original.
# train-yolov5-face
