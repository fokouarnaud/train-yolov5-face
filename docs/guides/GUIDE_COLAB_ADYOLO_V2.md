# ADYOLOv5-Face - Guide Google Colab (Version Optimisée)

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

```python
# Test de validation ADYOLOv5-Face
!python test_adyolo_colab.py
```

### 📂 Structure des Fichiers sur Google Drive

```
/content/drive/MyDrive/yolov5_face_scripts/
├── main.py
├── data_preparation.py
├── model_training.py
├── model_evaluation.py
├── utils.py
├── colab_setup.py          # ✨ VERSION OPTIMISÉE - Pas de duplication
├── config.py
└── test_adyolo_colab.py    # ✨ NOUVEAU - Test de validation
```

### 🐍 **Architecture de Maintenance Simplifiée**

**Approche précédente (problématique) :**
```
Repo local (modifications) + colab_setup.py (mêmes modifications) = Duplication
```

**Nouvelle approche (optimisée) :**
```
Repo local (modifications) → GitHub → Colab clone → Vérification seulement
```

**Avantages :**
- ✅ **Une seule source de vérité** : Toutes les modifications dans le repo local
- ✅ **Pas de duplication** : `colab_setup.py` ne modifie plus les fichiers
- ✅ **Maintenance simplifiée** : Un seul endroit à mettre à jour
- ✅ **Moins d'erreurs** : Pas de risque de versions différentes

### ⚡ Changements Clés dans les Scripts

#### 1. `colab_setup.py` - Version optimisée
- ✅ **Vérification seulement** : Check si les fichiers ADYOLOv5-Face sont présents
- ✅ **Pas de création** : N'essaie plus de créer `gd.py`, `yolo.py`, etc.
- ✅ **Maintenance simplifiée** : Une seule source de vérité (repo local)
- ✅ **Validation automatique** : Test après setup

#### 2. `test_adyolo_colab.py` - Nouveau script
- ✅ Test d'importation sans dépendance circulaire
- ✅ Validation de l'architecture ADYOLOv5-Face
- ✅ Test de forward pass avec 4 têtes de détection

### 🎯 **Qu'est-ce qui a été Résolu ?**

#### Problème d'Importation Circulaire
**Avant :**
```
ImportError: cannot import name 'Conv' from partially initialized module 'models.common'
```

**Après :**
```
✅ Tous les modules s'importent correctement
✅ GDFusion fonctionne avec sa propre classe Conv
✅ Plus de dépendance circulaire
```

#### Problème de Maintenance
**Avant :**
```
❌ Modifications à faire dans 2 endroits :
   - Repo local (gd.py, yolo.py, etc.)
   - colab_setup.py (mêmes fichiers)
```

**Après :**
```
✅ Modifications uniquement dans le repo local
✅ colab_setup.py vérifie seulement la présence
✅ Une seule source de vérité
```

### 🚀 Résultats Attendus

**Après `!python colab_setup.py --model-size ad` :**
```
=== Vérification d'ADYOLOv5-Face ===
✓ gd.py présent
✓ adyolov5s_simple.yaml présent  
✓ hyp.adyolo.yaml présent
✓ Tous les fichiers ADYOLOv5-Face sont présents

🎉 ALL TESTS PASSED!
✅ ADYOLOv5-Face is ready for training on Google Colab!
✅ No circular import issues!
✅ PyTorch 2.6+ compatible!
```

**Après `!python main.py --model-size ad` :**
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

**Si vous rencontrez des erreurs :**

1. **"Fichiers manquants pour ADYOLOv5-Face"**
   - Le repo GitHub n'a pas été mis à jour avec les modifications locales
   - Solution : Pusher les modifications locales vers GitHub

2. **Erreur d'importation**
   - Exécutez `!python test_adyolo_colab.py` pour diagnostiquer
   - Vérifiez que le repo contient les modifications ADYOLOv5-Face

3. **"Modèle non trouvé"**
   - Vérifiez que `--model-size ad` est utilisé
   - Le fichier `adyolov5s_simple.yaml` doit être présent

4. **"CUDA out of memory"**
   - Réduisez `--batch-size` dans `main.py`
   - Utilisez des images plus petites

### 🎉 Workflow Final Optimisé

```
1. Modifications locales (repo yolov5-face)
   ↓
2. Push vers GitHub
   ↓
3. Colab clone le repo mis à jour
   ↓
4. colab_setup.py vérifie la présence des fichiers
   ↓  
5. Entraînement ADYOLOv5-Face
```

**Plus de duplication, plus de problèmes de maintenance !** ✨

La solution est maintenant **robuste** et **facile à maintenir** ! 🎉
