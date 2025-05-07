# YOLOv5-Face Trainer pour Google Colab

Ce projet est une implémentation structurée et corrigée pour l'entraînement du modèle YOLOv5-Face sur Google Colab, spécialisé dans la détection de visages en utilisant le dataset WIDER Face. Il inclut des correctifs importants pour résoudre les problèmes de compatibilité avec les versions récentes de PyTorch (2.6+).

## Structure du projet

Le projet est organisé en plusieurs modules pour faciliter la maintenance et la compréhension :

- `main.py` : Script principal pour exécuter l'ensemble du pipeline
- `data_preparation.py` : Gestion de la préparation des données WIDER Face
- `model_training.py` : Entraînement du modèle YOLOv5-Face
- `model_evaluation.py` : Évaluation et exportation du modèle
- `utils.py` : Fonctions utilitaires communes
- `colab_setup.py` : Configuration de l'environnement et clonage du dépôt
- `pytorch_fix.py` : Correction de compatibilité pour PyTorch 2.6+
- `plan.txt` : Documentation de l'état du projet et des améliorations

## Prérequis

- Google Colab avec GPU activé
- Accès à Google Drive avec les fichiers du dataset WIDER Face :
  - `/content/drive/MyDrive/dataset/WIDER_train.zip`
  - `/content/drive/MyDrive/dataset/WIDER_val.zip`
  - `/content/drive/MyDrive/dataset/WIDER_test.zip`
  - `/content/drive/MyDrive/dataset/retinaface_gt.zip`

## Installation

1. Créez un dossier dans votre Google Drive pour stocker les scripts :
   `/content/drive/MyDrive/yolov5_face_scripts/`

2. Téléchargez et placez tous les fichiers du projet dans ce dossier
   
3. Créez un dossier pour les datasets :
   `/content/drive/MyDrive/dataset/`
   
4. Placez les fichiers WIDER Face dans ce dossier

## Utilisation

### Méthode recommandée

Suivez ces étapes dans un notebook Google Colab :

```python
# Étape 1: Monter Google Drive et copier les scripts
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content
!cp /content/drive/MyDrive/yolov5_face_scripts/{main.py,data_preparation.py,model_training.py,model_evaluation.py,utils.py,colab_setup.py,pytorch_fix.py} /content/

# Étape 2: Installer les dépendances compatibles
!pip install numpy==1.26.4 scipy==1.13.1 gensim==4.3.3 --no-deps
!pip install torch>=2.0.0 torchvision>=0.15.0

# Étape 3: Exécuter le script de configuration
%cd /content
!python colab_setup.py --model-size s

# Étape 3.5: Corriger la compatibilité PyTorch
!python pytorch_fix.py

# Étape 4: Lancer l'entraînement
!python main.py

# Étape 5: Visualiser les résultats
%load_ext tensorboard
%tensorboard --logdir /content/yolov5-face/runs/train/face_detection_transfer
```

### Options avancées pour l'entraînement

Vous pouvez personnaliser l'exécution avec différentes options :

```python
# Exemples d'utilisation avec des options personnalisées
!python main.py --batch-size 16 --epochs 100 --img-size 800 --model-size m
```

Options disponibles :
- `--batch-size` : Taille du batch pour l'entraînement (défaut: 32)
- `--epochs` : Nombre d'epochs d'entraînement (défaut: 300)
- `--img-size` : Taille d'image pour l'entraînement (défaut: 640)
- `--model-size` : Taille du modèle YOLOv5 (s, m, l, x) (défaut: s)
- `--yolo-version` : Version de YOLOv5 à utiliser (défaut: 5.0)
- `--skip-train` : Ignorer l'étape d'entraînement
- `--skip-evaluation` : Ignorer l'étape d'évaluation
- `--skip-export` : Ignorer l'étape d'exportation

## Vérifications recommandées

Pour vous assurer que tout est correctement configuré :

```python
# Vérifier la présence des fichiers nécessaires
import os
required_files = ['main.py', 'data_preparation.py', 'model_training.py', 'model_evaluation.py', 'utils.py', 'pytorch_fix.py']
missing_files = [f for f in required_files if not os.path.exists(f'/content/{f}')]
if missing_files:
    print(f"⚠️ Fichiers manquants: {', '.join(missing_files)}")
else:
    print("✅ Tous les fichiers nécessaires sont présents.")

# Vérifier que le dépôt YOLOv5-Face est correctement cloné
if os.path.exists('/content/yolov5-face'):
    print("✅ Le dépôt YOLOv5-Face est correctement cloné.")
else:
    print("⚠️ Le dépôt YOLOv5-Face n'est pas cloné.")

# Vérifier que la correction PyTorch a été appliquée
!grep "weights_only=False" /content/yolov5-face/train.py
```

## Corrections apportées

Cette version corrige plusieurs problèmes du script original :

1. **Problème API NumPy** : Correction de l'erreur liée à `np.int` déprécié (remplacé par `np.int32`)
2. **Compatibilité PyTorch 2.6+** : Ajout du paramètre `weights_only=False` à la fonction `torch.load()`
3. **Gestion des images corrompues** : Amélioration du filtrage des images et annotations non valides
4. **Structure du code** : Organisation en modules pour une meilleure maintenabilité
5. **Robustesse des commandes** : Utilisation de `subprocess.run()` au lieu de commandes shell directes
6. **Gestion des erreurs** : Meilleure gestion des exceptions et messages d'erreur détaillés

## Dépannage

Si vous rencontrez cette erreur : 
```
ckpt = torch.load(weights, map_location=device)  # load checkpoint
```

Exécutez le script de correction PyTorch :
```python
!python pytorch_fix.py
```

Ce script corrige automatiquement les problèmes de compatibilité avec PyTorch 2.6+, y compris les situations où une correction incorrecte a été appliquée précédemment.

## Résultats

Après l'exécution complète, vous trouverez les résultats dans les répertoires suivants :

- Modèle PyTorch : `/content/yolov5-face/runs/train/face_detection_transfer/weights/best.pt`
- Modèle ONNX : `/content/yolov5-face/runs/train/face_detection_transfer/weights/best.onnx`
- Métriques et logs : `/content/yolov5-face/runs/train/face_detection_transfer`
- Visualisation TensorBoard : Accessible via TensorBoard dans Colab

## Sauvegarde des résultats

Pour sauvegarder les résultats sur votre Google Drive :

```python
# Créer un dossier pour les résultats
!mkdir -p /content/drive/MyDrive/YOLOv5_Face_Results

# Copier les résultats de l'entraînement vers Google Drive
!cp -r /content/yolov5-face/runs/train/face_detection_transfer /content/drive/MyDrive/YOLOv5_Face_Results/
```

## Licence

Ce projet est basé sur [YOLOv5-Face](https://github.com/deepcam-cn/yolov5-face) et suit la même licence que le projet original.
