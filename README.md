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
- `fix_loss_py.py` : Correction des problèmes de conversion de type dans loss.py
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

# Étape 1: Monter Google Drive et copier les scripts
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content
!cp /content/drive/MyDrive/yolov5_face_scripts/{main.py,data_preparation.py,model_training.py,model_evaluation.py,utils.py,colab_setup.py} /content/

# Étape 2: Installer les dépendances compatibles
!pip install numpy==1.26.4 scipy==1.13.1 gensim==4.3.3 --no-deps
!pip install torch>=2.0.0 torchvision>=0.15.0
!pip install opencv-python werkzeug

# Étape 3: Exécuter le script de configuration
%cd /content
!python colab_setup.py --model-size s

# Étape 4: Lancer l'entraînement
!python main.py

# Étape 5: Visualiser les résultats
%load_ext tensorboard
%tensorboard --logdir /content/yolov5-face/runs/train/face_detection_transfer

# Étape 6: Sauvegarder les résultats
!mkdir -p /content/drive/MyDrive/YOLOv5_Face_Results
!cp -r /content/yolov5-face/runs/train/face_detection_transfer /content/drive/MyDrive/YOLOv5_Face_Results/

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
required_files = ['main.py', 'data_preparation.py', 'model_training.py', 'model_evaluation.py', 'utils.py', 'pytorch_fix.py', 'fix_loss_py.py']
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

# Vérifier que la correction loss.py a été appliquée
!grep "long()" /content/yolov5-face/utils/loss.py
```

## Corrections apportées

Cette version corrige plusieurs problèmes du script original :

1. **Problème API NumPy** : Correction de l'erreur liée à `np.int` déprécié (remplacé par `np.int32`)
2. **Compatibilité PyTorch 2.6+** : Ajout du paramètre `weights_only=False` à la fonction `torch.load()`
3. **Problème de conversion de type dans loss.py** : Ajout de la méthode `.long()` aux résultats de `clamp_()`
4. **Gestion des images corrompues** : Amélioration du filtrage des images et annotations non valides
5. **Structure du code** : Organisation en modules pour une meilleure maintenabilité
6. **Robustesse des commandes** : Utilisation de `subprocess.run()` au lieu de commandes shell directes
7. **Gestion des erreurs** : Meilleure gestion des exceptions et messages d'erreur détaillés

## Dépannage

Si vous rencontrez cette erreur dans `train.py` : 
```
ckpt = torch.load(weights, map_location=device)  # load checkpoint
```

Exécutez le script de correction PyTorch :
```python
!python pytorch_fix.py
```

Si vous rencontrez cette erreur dans `face_datasets.py` :
```
bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
AttributeError: module 'numpy' has no attribute 'int'.
```

Vérifiez que `utils.py` est bien le fichier mis à jour avec la correction améliorée pour `fix_numpy_issue()`.

Si vous rencontrez cette erreur dans `loss.py` :
```
indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))
RuntimeError: result type Float can't be cast to the desired output type long int
```

Exécutez le script de correction pour loss.py :
```python
!python fix_loss_py.py
```

Si vous rencontrez une erreur `ModuleNotFoundError: No module named 'cv2'` :
```python
!pip install opencv-python
```

Pour les problèmes liés à TensorBoard :
```python
!pip install tensorboard
%load_ext tensorboard
```

## Optimisation pour différents accélérateurs matériels

### GPU T4 (Google Colab standard)
Configuration optimale par défaut pour le T4 GPU (recommandée pour la plupart des utilisateurs) :
```python
!python main.py --batch-size 16 --img-size 640 --model-size s
```

### TPU v2-8
Si vous avez accès aux TPUs v2-8 (nécessite des ajustements pour l'utilisation des TPUs) :
```python
# Note: L'utilisation de TPUs nécessite des modifications supplémentaires au code
!python main.py --batch-size 32 --img-size 640 --model-size s
```

### TPU v5e-1 / v6e-1
Pour les versions plus récentes des TPUs :
```python
# Note: L'utilisation de TPUs nécessite des modifications supplémentaires au code
!python main.py --batch-size 64 --img-size 640 --model-size m
```

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

## Contributions et améliorations futures

Les contributions à ce projet sont les bienvenues. Voici quelques améliorations planifiées :

1. Intégration de toutes les corrections dans un seul script pour simplifier le processus
2. Support amélioré pour les accélérateurs matériels TPU
3. Implémentation d'un système de sauvegarde automatique des résultats sur Google Drive
4. Optimisation des hyperparammètres pour différentes tailles de modèles

## Licence

Ce projet est basé sur [YOLOv5-Face](https://github.com/deepcam-cn/yolov5-face) et suit la même licence que le projet original.
