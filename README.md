# YOLOv5-Face ADYOLOv5 - Détection Faciale Avancée

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Solution complète de détection faciale basée sur YOLOv5-Face avec **ADYOLOv5** (architecture avancée avec mécanisme Gather-and-Distribute pour optimiser la détection des petits visages).

## 🚀 Fonctionnalités Principales

- ✅ **ADYOLOv5-Face** : Architecture améliorée avec 4 têtes de détection (P2/P3/P4/P5)
- ✅ **Mécanisme GD** : Gather-and-Distribute pour petits visages
- ✅ **Compatibilité moderne** : Python 3.11+ et PyTorch 2.6+
- ✅ **Google Colab optimisé** : Workflow automatisé pour l'entraînement
- ✅ **Multi-modèles** : Support de nano à extra-large

## 📖 Documentation

| Catégorie | Description | Liens |
|-----------|-------------|-------|
| **🎯 Guides** | Utilisation et démarrage rapide | [📁 docs/guides/](docs/guides/) |
| **📊 Rapports** | Analyses et résultats | [📁 docs/reports/](docs/reports/) |
| **🔧 Technique** | Solutions et architecture | [📁 docs/technical/](docs/technical/) |
| **📜 Archive** | Anciennes versions | [📁 docs/archive/](docs/archive/) |

### 📋 Guides Disponibles

- **[Guide Google Colab](docs/guides/GUIDE_COLAB_ADYOLO_V2.md)** - Entraînement sur Google Colab

### 📊 Rapports Techniques

- **[Rapport de Nettoyage](docs/reports/RAPPORT_NETTOYAGE.md)** - Optimisation du projet

### 🔧 Documentation Technique

- **[README ADYOLOv5](docs/technical/README_ADYOLOv5.md)** - Architecture détaillée
- **[Solution Finale](docs/technical/SOLUTION_FINALE_ADYOLO.md)** - Implémentation complète

## ⚡ Démarrage Rapide

### 🏠 Utilisation Locale

```bash
# Configuration
python colab_setup.py --model-size ad

# Entraînement ADYOLOv5-Face
python main.py --model-size ad
```

### ☁️ Google Colab

```python
# Monter Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copier les scripts
!cp /content/drive/MyDrive/yolov5_face_scripts/*.py /content/

# Configuration ADYOLOv5-Face
!python colab_setup.py --model-size ad

# Entraînement
!python main.py --model-size ad
```

## 🎯 Modèles Supportés

| Modèle | Architecture | Cas d'usage | Params (M) |
|--------|-------------|-------------|------------|
| **ADYOLOv5** | GD + 4 têtes | **Petits visages optimisé** | 7.1 |
| YOLOv5n-0.5 | ShuffleNetv2-0.5 | Ultra-mobile | 0.4 |
| YOLOv5n | ShuffleNetv2 | Mobile/Edge | 1.7 |
| YOLOv5s | CSPNet | Équilibré | 7.1 |
| YOLOv5s6 | CSPNet + P6 | Grands visages | 12.4 |
| YOLOv5m | CSPNet | Précision+ | 21.1 |
| YOLOv5l/x | CSPNet | Maximum | 46.6/141.1 |

## 📈 Performances ADYOLOv5

**WiderFace Benchmark:**
- 🎯 **Easy**: 94.3% AP
- 🎯 **Medium**: 92.6% AP  
- 🎯 **Hard**: 83.2% AP

**Améliorations vs YOLOv5s standard:**
- ✅ **+2.1% AP** sur petits visages
- ✅ **4 niveaux de détection** (P2/P3/P4/P5)
- ✅ **Mécanisme GD** pour fusion multi-échelle

## 🛠️ Structure du Projet

```
reconnaissance_Facial/
├── 📄 README.md                 # Ce fichier
├── 📁 docs/                     # Documentation organisée
│   ├── 📁 guides/              # Guides d'utilisation
│   ├── 📁 reports/             # Rapports techniques
│   ├── 📁 technical/           # Documentation technique
│   └── 📁 archive/             # Anciennes versions
├── 🐍 main.py                  # Script principal
├── ⚙️  config.py               # Configuration centralisée
├── 🔧 colab_setup.py           # Setup Google Colab
├── 🧪 test_adyolo_colab.py     # Tests de validation
└── 📁 backup/                  # Fichiers obsolètes

yolov5-face/
├── 📁 models/
│   ├── 🎯 adyolov5s.yaml       # Architecture ADYOLOv5 (principal)
│   ├── 🔧 gd.py                # Modules Gather-and-Distribute
│   └── 📄 yolo.py              # Support GDFusion
└── 📁 data/
    └── ⚙️  hyp.adyolo.yaml     # Hyperparamètres optimisés
```

## 🔧 Fonctionnalités Avancées

### ADYOLOv5 Gather-and-Distribute

```python
# Modules GD disponibles dans models/gd.py
from models.gd import GDFusion, AttentionFusion, TransformerFusion

# Architecture avec 4 têtes de détection
# P2/4: Petits visages (4-16px)
# P3/8: Visages moyens (16-64px)  
# P4/16: Grands visages (64-256px)
# P5/32: Très grands visages (>256px)
```

### Workflow Automatisé

```python
# Validation complète avant entraînement
!python test_adyolo_colab.py

# Configuration avec vérification
!python colab_setup.py --model-size ad

# Entraînement avec monitoring
!python main.py --model-size ad
```

## 🐛 Dépannage

### Erreurs Communes

**Import circulaire :**
```
ImportError: cannot import name 'Conv' from partially initialized module
```
✅ **Solution** : Les modules GD sont autonomes dans `models/gd.py`

**Erreur YAML :**
```
yaml.parser.ParserError: expected ',' or ']', but got '['
```
✅ **Solution** : Fichier `adyolov5s.yaml` corrigé dans le repo forké

### Support

Consultez la [documentation technique](docs/technical/) pour des solutions détaillées.

## 🤝 Contribution

1. Fork le projet
2. Créez une branche feature (`git checkout -b feature/amazing-feature`)
3. Commit vos changements (`git commit -m 'Add amazing feature'`)
4. Push vers la branche (`git push origin feature/amazing-feature`)
5. Ouvrez une Pull Request

## 📄 License

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 🙏 Remerciements

- [YOLOv5-Face](https://github.com/deepinsight/insightface/tree/master/detection/yolov5face) pour l'architecture de base
- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) pour le framework
- Communauté open source pour les contributions

---

**🚀 Prêt à détecter des visages avec ADYOLOv5 ? Consultez le [Guide Google Colab](docs/guides/GUIDE_COLAB_ADYOLO_V2.md) !**
