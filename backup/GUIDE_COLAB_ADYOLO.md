# ADYOLOv5-Face - Guide Google Colab

## Commandes à Exécuter sur Google Colab

### 🚀 Setup Complet (Une seule fois)

```python
# YOLOv5-Face: Entraînement et Évaluation ADYOLOv5-Face
# Utilise le repo modifié manuellement avec résolution d'importation circulaire

# Étape 1: Monter Google Drive et copier les scripts
from google.colab import drive
drive.mount('/content/drive')

# Copier les scripts depuis Drive (MISE À JOUR avec test_adyolo_colab.py)
!mkdir -p /content
!cp /content/drive/MyDrive/yolov5_face_scripts/main.py \
   /content/drive/MyDrive/yolov5_face_scripts/data_preparation.py \
   /content/drive/MyDrive/yolov5_face_scripts/model_training.py \
   /content/drive/MyDrive/yolov5_face_scripts/model_evaluation.py \
   /content/drive/MyDrive/yolov5_face_scripts/utils.py \
   /content/drive/MyDrive/yolov5_face_scripts/colab_setup.py \
   /content/drive/MyDrive/yolov5_face_scripts/config.py \
   /content/drive/MyDrive/yolov5_face_scripts/test_adyolo_colab.py /content/

# Étape 2: Installer les dépendances compatibles
!pip install numpy==1.26.4 scipy==1.13.1 gensim==4.3.3 --no-deps
!pip install torch>=2.0.0 torchvision>=0.15.0
!pip install opencv-python
!pip install --upgrade nvidia-cudnn-cu11 nvidia-cublas-cu11
!pip install werkzeug  # Pour TensorBoard

# Étape 3: Exécuter le script de configuration ADYOLOv5-Face
%cd /content
!python colab_setup.py --model-size ad

# Étape 4: Lancer l'entraînement et l'évaluation ADYOLOv5-Face
!python main.py --model-size ad

# Étape 5: Visualiser les résultats
%load_ext tensorboard
%tensorboard --logdir /content/yolov5-face/runs/train/face_detection_transfer
```

### 🔍 Tests de Validation (Optionnel)

Si vous voulez tester séparément :

```python
# Test de validation ADYOLOv5-Face
!python test_adyolo_colab.py
```

### 📂 Structure des Fichiers sur Google Drive

Assurez-vous que votre Google Drive contient ces fichiers mis à jour :

```
/content/drive/MyDrive/yolov5_face_scripts/
├── main.py
├── data_preparation.py
├── model_training.py
├── model_evaluation.py
├── utils.py
├── colab_setup.py          # ✨ MISE À JOUR - Support ADYOLOv5-Face
├── config.py
└── test_adyolo_colab.py    # ✨ NOUVEAU - Test de validation
```

### ⚡ Changements Clés dans les Scripts

#### 1. `colab_setup.py` - Version optimisée
- ✅ Vérification de la présence des fichiers ADYOLOv5-Face
- ✅ Pas de duplication de code (fichiers déjà dans le repo)
- ✅ Validation automatique après setup
- ✅ Maintenance simplifiée

#### 2. `test_adyolo_colab.py` - Nouveau script
- ✅ Test d'importation sans dépendance circulaire
- ✅ Validation de l'architecture ADYOLOv5-Face
- ✅ Test de forward pass avec 4 têtes de détection

### 🎯 Qu'est-ce qui a été Résolu ?

**Problème Original :**
```
ImportError: cannot import name 'Conv' from partially initialized module 'models.common'
```

**Solution Implémentée :**
1. **Module GD autonome** - `models/gd.py` avec sa propre classe `Conv`
2. **Suppression d'importation circulaire** - Plus d'import GD dans `common.py`
3. **Architecture ADYOLOv5-Face complète** - 4 têtes P2/P3/P4/P5 avec mécanisme Gather-and-Distribute

### 🚀 Résultats Attendus

Après `!python colab_setup.py --model-size ad` :
```
🎉 ALL TESTS PASSED!
✅ ADYOLOv5-Face is ready for training on Google Colab!
✅ No circular import issues!
✅ PyTorch 2.6+ compatible!
```

Après `!python main.py --model-size ad` :
- Entraînement ADYOLOv5-Face avec 4 têtes de détection
- Mécanisme Gather-and-Distribute actif
- Modèles exportés en PyTorch (.pt) et ONNX (.onnx)

### 📊 Architecture ADYOLOv5-Face

```
Input (640x640)
    ↓
Backbone (Focus + C3 + SPP)
    ↓
Gather-and-Distribute Mechanism
├── Low-Stage GD (Attention Fusion)
└── High-Stage GD (Transformer Fusion)
    ↓
4 Detection Heads
├── P2/4 (Small faces)
├── P3/8 (Medium faces)  
├── P4/16 (Large faces)
└── P5/32 (Extra large faces)
```

### 🔧 Dépannage

Si vous rencontrez des erreurs :

1. **Erreur d'importation** : Exécutez `!python test_adyolo_colab.py`
2. **Modèle non trouvé** : Vérifiez que `--model-size ad` est utilisé
3. **CUDA out of memory** : Réduisez `--batch-size` dans `main.py`

La solution est maintenant complète et prête pour Google Colab ! 🎉
