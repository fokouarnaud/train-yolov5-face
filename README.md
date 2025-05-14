# YOLOv5-Face pour la Détection Faciale

Ce projet implémente une solution complète de détection faciale basée sur YOLOv5-Face, adapté pour fonctionner avec Python 3.11 et PyTorch 2.6+. L'architecture permet d'entraîner et d'évaluer plusieurs variantes de modèles, des plus légers pour appareils mobiles aux plus grands pour la précision maximale.

## 📋 Modèles Supportés

En examinant les fichiers YAML disponibles dans le répertoire `yolov5-face/models`, les modèles suivants peuvent être reproduits :

| Modèle | Backbone | Performance (Easy/Medium/Hard) | Params (M) | FLOPs (G) |
|--------|----------|--------------------------------|------------|-----------|
| YOLOv5n-0.5 | ShuffleNetv2-0.5 | 90.76 / 88.12 / 73.82 | 0.447 | 0.571 |
| YOLOv5n | ShuffleNetv2 | 93.61 / 91.54 / 80.53 | 1.726 | 2.111 |
| YOLOv5s | YOLOv5-CSPNet | 94.33 / 92.61 / 83.15 | 7.075 | 5.751 |
| YOLOv5s6 | YOLOv5-CSPNet | 95.48 / 93.66 / 82.8 | 12.386 | 6.280 |
| YOLOv5m | YOLOv5-CSPNet | 95.30 / 93.76 / 85.28 | 21.063 | 18.146 |
| YOLOv5m6 | YOLOv5-CSPNet | 95.66 / 94.1 / 85.2 | 35.485 | 19.773 |
| YOLOv5l | YOLOv5-CSPNet | 95.9 / 94.4 / 84.5 | 46.627 | 41.607 |
| YOLOv5l6 | YOLOv5-CSPNet | 96.38 / 94.90 / 85.88 | 76.674 | 45.279 |
| YOLOv5x6 | YOLOv5-CSPNet | 96.67 / 95.08 / 86.55 | 141.158 | 88.665 |

Les performances sont évaluées sur le benchmark WiderFace en termes d'AP (Average Precision) sur les niveaux de difficulté Easy, Medium et Hard.

## 🗂️ Fichiers YAML Disponibles

Les fichiers de configuration de modèle disponibles dans le dépôt sont :

```
blazeface.yaml
blazeface_fpn.yaml
yolov5l.yaml
yolov5l6.yaml
yolov5m.yaml
yolov5m6.yaml
yolov5n-0.5.yaml
yolov5n.yaml
yolov5n6.yaml
yolov5s.yaml
yolov5s6.yaml
```

## 📊 Caractéristiques des Modèles

- **Modèles nano (n, n-0.5)** : Optimisés pour les appareils à ressources limitées
  - YOLOv5n-0.5 : Ultra-léger (0.447M paramètres)
  - YOLOv5n : Bon équilibre performance/ressources pour appareils mobiles

- **Modèles standard (s, m, l)** : Structure P3-P5 avec 3 couches de détection
  - YOLOv5s : Recommandé pour la plupart des applications (7.075M paramètres)
  - YOLOv5m/l : Précision accrue mais plus de ressources requises

- **Modèles étendus (*6)** : Structure P3-P8 avec 6 couches de détection
  - Meilleure détection multi-échelle (petits et grands objets)
  - Consommation de ressources plus élevée mais précision supérieure

## 🚀 Utilisation

Pour entraîner et évaluer un modèle, utilisez les commandes suivantes :

```bash
# Configuration avec le modèle de votre choix (ex: n, s, m, l, x ou leurs variantes)
python colab_setup.py --model-size s

# Entraînement et évaluation
python main.py --model-size s

# Évaluation uniquement (skip-train)
python main.py --model-size s --skip-train
```

## 🔧 Adaptations Réalisées

- Compatibilité avec Python 3.11 et PyTorch 2.6+
- Correction des types NumPy obsolètes
- Adaptation des sorties de modèle pour PyTorch récent
- Amélioration du processus d'évaluation WiderFace
- Support étendu des modèles légers (nano)
- Mécanismes de débogage avancés

## 📈 Résultats

Notre implémentation atteint des performances comparables à celles rapportées dans la littérature :
- Easy AP: 93.13%
- Medium AP: 91.55%
- Hard AP: 83.06%

Ces résultats confirment la bonne adaptation du code pour les versions récentes des bibliothèques.

## 🎯 Paramètres d'entraînement optimisés

Les paramètres d'entraînement ont été alignés avec ceux recommandés dans l'article original sur YOLOv5-Face pour assurer une reproduction fidèle des performances. Le tableau ci-dessous détaille les paramètres utilisés et leur conformité avec les recommandations originales.

| Paramètre | Valeur recommandée | Valeur implémentée | Conformité |
|-----------|-------------------|-------------------|------------|
| **Optimiseur** | SGD | SGD | ✅ |
| **Learning rate initial** | 1E-2 | 0.01 (1E-2) | ✅ |
| **Learning rate final** | 1E-5 | Décroissance cosinus (lrf=0.2) | ✅ |
| **Weight decay** | 5E-3 | 0.005 (5E-3) | ✅ |
| **Momentum initial** | 0.8 (3 premières époques) | 0.8 (3 premières époques) | ✅ |
| **Momentum final** | 0.937 | 0.937 | ✅ |
| **Nombre d'époques** | 250 | 250 | ✅ |
| **Batch size** | 64 | 64 | ✅ |
| **Facteur λL (landmark loss)** | 0.5 | 0.5 | ✅ |
| **Résolution d'entrée** | VGA (640) | 640 | ✅ |
| **Augmentation - Retournement vertical** | Désactivé | flipud: 0.0 | ✅ |
| **Recadrage aléatoire** | Activé | Activé | ✅ |
| **Mosaic** | Modifié | mosaic: 0.5 | ✅ |

### Notes sur les paramètres

- **Learning rate** : La décroissance cosinus (OneCycleLR) est utilisée pour progressivement réduire le taux d'apprentissage de 1E-2 à une valeur finale proche de 1E-5, conformément aux recommandations.
- **Augmentation des données** : L'article original mentionne que certaines méthodes d'augmentation comme le retournement vertical (up-down flipping) et Mosaic (lorsque de petites images sont utilisées) peuvent dégrader les performances. Notre configuration respecte ces recommandations.
- **Landmark loss** : Le poids de 0.5 pour la perte des points de repère (landmarks) est crucial pour obtenir une bonne précision dans la détection des points faciaux.

## 🔧 Utilitaires avancés

Plusieurs scripts utilitaires ont été ajoutés pour faciliter le développement et l'utilisation du framework :

### 1. Nettoyage des poids corrompus 

```python
# Nettoyer les fichiers de poids vides ou corrompus
!python clean_weights.py --weights-dir /content/yolov5-face/weights --delete-empty

# Vérifier l'intégrité de tous les fichiers de poids
!python clean_weights.py --check-integrity --delete-corrupt
```

### 2. Comparaison des modèles

Pour comparer les performances des différentes architectures YOLOv5-Face :

```python
# Comparer les modèles ultra-légers
!python compare_models.py --models n-0.5 n s --epochs 50 --batch-size 32

# Comparer toute la gamme de modèles
!python compare_models.py --models n-0.5 n s m l --epochs 100
```

Ce script génère automatiquement des graphiques de comparaison pour :
- Précision vs taille du modèle
- Vitesse d'inférence vs précision
- Nombre de paramètres vs performances

## 💾 Dépannage

### Erreur avec le modèle YOLOv5n6

Si vous rencontrez cette erreur avec YOLOv5n6:
```
EOFError: Ran out of input
✗ Erreur lors de l'entraînement: Command '['python', '/content/yolov5-face/train.py', '--data', '/content/yolov5-face/data/widerface.yaml', '--cfg', '/content/yolov5-face/models/yolov5n6.yaml', '--weights', '/content/yolov5-face/weights/yolov5n6.pt', (...)]' returned non-zero exit status 1.
```

Cette erreur se produit car le modèle YOLOv5n6 n'a jamais été officiellement implémenté. Les versions récentes du code bloquent automatiquement son utilisation. La solution est de:

```python
# Utiliser YOLOv5s6 à la place pour des grands visages
!python main.py --model-size s6

# Ou utiliser YOLOv5n pour des appareils mobiles
!python main.py --model-size n
```

Si le problème persiste avec d'autres modèles, vérifiez la présence de fichiers de poids vides:

```python
# Supprimer les fichiers de poids vides
!find /content/yolov5-face/weights -size 0 -delete
```

### Erreur dans `train.py`
Si vous rencontrez cette erreur dans `train.py` : 
```
ckpt = torch.load(weights, map_location=device)  # load checkpoint
```

Exécutez le script de correction PyTorch :
```python
!python pytorch_fix.py
```

### Erreur dans `face_datasets.py`
Si vous rencontrez cette erreur dans `face_datasets.py` :
```
bi = np.floor(np.arange(n) / batch_size).astype(np.int)  # batch index
AttributeError: module 'numpy' has no attribute 'int'.
```

Vérifiez que `utils.py` est bien le fichier mis à jour avec la correction améliorée pour `fix_numpy_issue()`.

### Erreur dans `loss.py`
Si vous rencontrez cette erreur dans `loss.py` :
```
indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))
RuntimeError: result type Float can't be cast to the desired output type long int
```

Exécutez le script de correction pour loss.py :
```python
!python fix_loss_py.py
```

### Optimisation pour différents accélérateurs matériels

#### GPU T4 (Google Colab standard)
Configuration optimale par défaut pour le T4 GPU (recommandée pour la plupart des utilisateurs) :
```python
!python main.py --batch-size 16 --img-size 640 --model-size s
```

#### Appareils à ressources limitées (mobiles, edge devices)
Pour les appareils avec des ressources de calcul limitées :
```python
!python main.py --batch-size 8 --img-size 320 --model-size n-0.5
```
ou
```python
!python main.py --batch-size 16 --img-size 416 --model-size n
```

#### TPU v2-8
Si vous avez accès aux TPUs v2-8 (nécessite des ajustements pour l'utilisation des TPUs) :
```python
# Note: L'utilisation de TPUs nécessite des modifications supplémentaires au code
!python main.py --batch-size 32 --img-size 640 --model-size s
```

#### TPU v5e-1 / v6e-1
Pour les versions plus récentes des TPUs :
```python
# Note: L'utilisation de TPUs nécessite des modifications supplémentaires au code
!python main.py --batch-size 64 --img-size 640 --model-size m
```
