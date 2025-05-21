# ADYOLOv5-Face: Détecteur de visages amélioré avec mécanisme Gather-and-Distribute

Ce projet intègre ADYOLOv5-Face au framework YOLOv5-Face. ADYOLOv5-Face est une version améliorée qui utilise un mécanisme Gather-and-Distribute (GD) et une tête de détection supplémentaire pour améliorer significativement la détection des petits visages.

## Sommaire

- [Présentation](#présentation)
- [Caractéristiques principales](#caractéristiques-principales)
- [Architecture](#architecture)
- [Performances](#performances)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Entraînement sur Google Colab](#entraînement-sur-google-colab)
- [Exportation du modèle](#exportation-du-modèle)
- [Intégration avec l'application Flutter](#intégration-avec-lapplication-flutter)
- [Références](#références)

## Présentation

ADYOLOv5-Face est une version améliorée de YOLOv5s-Face spécifiquement conçue pour améliorer la détection des petits visages dans des environnements complexes, comme les salles de classe et les scènes de foule. Développé sur la base de l'architecture YOLO (You Only Look Once), ce modèle apporte des améliorations significatives pour la détection des visages de petite taille, partiellement occultés ou dans des conditions d'éclairage difficiles.

## Caractéristiques principales

1. **Mécanisme Gather-and-Distribute (GD)**: Remplace la structure Feature Pyramid Network (FPN) + Path Aggregation Network (PAN) traditionnelle pour améliorer la fusion des caractéristiques entre les couches non adjacentes, réduisant ainsi la perte d'informations spatiales et sémantiques.

2. **Tête de détection supplémentaire pour petits visages**: Ajoute une quatrième tête de détection (P2/4) spécifique pour les petits visages, permettant une détection plus précise des visages de petite taille.

3. **Compatibilité PyTorch 2.6+**: Modifications pour assurer la compatibilité avec les versions récentes de PyTorch.

4. **Exportation ONNX optimisée**: Support pour l'exportation au format ONNX avec les optimisations nécessaires pour l'intégration mobile.

## Architecture

### Architecture globale d'ADYOLOv5-Face

L'architecture ADYOLOv5-Face se compose de trois parties principales:

1. **Backbone**: Extraction des caractéristiques visuelles (basé sur CSPDarknet)
2. **Neck**: Mécanisme GD pour la fusion multi-échelle des caractéristiques
3. **Head**: Quatre têtes de détection pour différentes tailles de visages

### Mécanisme Gather-and-Distribute détaillé

Le mécanisme GD comporte deux phases principales:

1. **Phase Gather (Rassemblement)**:
   - Collecte les caractéristiques des quatre échelles (P2, P3, P4, P5)
   - Les aligne et les fusionne pour créer une représentation globale
   - Utilise des couches `GatherLayer` spécialisées pour une fusion efficace

2. **Phase Distribute (Distribution)**:
   - Redistribue les informations fusionnées vers chaque niveau
   - Utilise des couches `DistributeLayer` pour maintenir la cohérence des caractéristiques
   - Préserve les informations spatiales et sémantiques à toutes les échelles

### Comparaison avec YOLOv5-Face standard

| Caractéristique | YOLOv5-Face | ADYOLOv5-Face |
|-----------------|-------------|---------------|
| Structure du cou | FPN + PAN | Mécanisme GD |
| Têtes de détection | 3 (P3, P4, P5) | 4 (P2, P3, P4, P5) |
| Fusion de caractéristiques | Couches adjacentes | Globale multi-échelle |
| Détection de petits visages | Limitée | Améliorée (tête P2/4) |

## Performances

ADYOLOv5-Face montre des améliorations significatives par rapport à YOLOv5s-Face sur le jeu de données WiderFace:

| Modèle | Easy (%) | Medium (%) | Hard (%) |
|--------|----------|------------|----------|
| YOLOv5s-Face | 94,33 | 92,61 | 83,15 |
| ADYOLOv5-Face | 94,80 | 93,77 | 84,37 |
| **Amélioration** | **+0,47** | **+1,16** | **+1,22** |

Les gains les plus importants sont observés dans les catégories "Medium" et "Hard" qui contiennent principalement des visages de petite taille ou partiellement occultés, démontrant l'efficacité des modifications apportées.

## Installation

### Prérequis

- Python 3.8 ou supérieur
- PyTorch 2.0+ (compatible avec PyTorch 2.6+)
- CUDA (recommandé pour l'entraînement)
- OpenCV

### Configuration locale

1. Cloner le dépôt YOLOv5-Face modifié:

```bash
git clone https://github.com/fokouarnaud/yolov5-face.git
cd yolov5-face
```

2. Installer les dépendances:

```bash
pip install numpy==1.26.4 scipy==1.13.1 gensim==4.3.3 --no-deps
pip install torch>=2.0.0 torchvision>=0.15.0
pip install opencv-python
pip install werkzeug  # Pour TensorBoard
```

## Utilisation

### Détection avec le modèle préentraîné

```python
import cv2
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords
import torch
import numpy as np

# Charger le modèle (utilisez la version standard qui implémente GatherLayer et DistributeLayer)
model = attempt_load('weights/adyolov5s.pt', map_location='cuda:0' if torch.cuda.is_available() else 'cpu')
model.eval()

# Préparer l'image
img = cv2.imread('test.jpg')
img0 = img.copy()
img = letterbox(img, new_shape=640)[0]
img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
img = np.ascontiguousarray(img)
img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
img = img.to('cuda:0' if torch.cuda.is_available() else 'cpu')

# Inférence
with torch.no_grad():
    pred = model(img)[0]
    
# Appliquer NMS
pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.5)

# Traiter les détections
for i, det in enumerate(pred):
    if len(det):
        # Redimensionner les coordonnées à l'image originale
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
        
        # Dessiner les résultats
        for *xyxy, conf, cls in det:
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            
            # Rectangle pour le visage
            cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Score de confiance
            label = f"{conf:.2f}"
            cv2.putText(img0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Points de repère faciaux (landmarks)
            if len(det[0]) > 15:  # Si les landmarks sont disponibles
                landmarks = det[0][5:15].view(-1, 2)
                for idx, (lx, ly) in enumerate(landmarks):
                    cv2.circle(img0, (int(lx), int(ly)), 2, (255, 0, 0), 2)

# Afficher le résultat
cv2.imshow('ADYOLOv5-Face Detection', img0)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Entraînement sur Google Colab

Pour entraîner ADYOLOv5-Face sur Google Colab:

1. Téléchargez les scripts dans votre Google Drive:
   - `main.py`
   - `data_preparation.py`
   - `model_training.py` 
   - `model_evaluation.py`
   - `utils.py`
   - `colab_setup.py`
   - `config.py`

2. Exécutez dans Google Colab:

```python
# Monter Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copier les scripts depuis Drive
!mkdir -p /content
!cp /content/drive/MyDrive/yolov5_face_scripts/main.py \
   /content/drive/MyDrive/yolov5_face_scripts/data_preparation.py \
   /content/drive/MyDrive/yolov5_face_scripts/model_training.py \
   /content/drive/MyDrive/yolov5_face_scripts/model_evaluation.py \
   /content/drive/MyDrive/yolov5_face_scripts/utils.py \
   /content/drive/MyDrive/yolov5_face_scripts/colab_setup.py \
   /content/drive/MyDrive/yolov5_face_scripts/config.py /content/

# Installer les dépendances
!pip install numpy==1.26.4 scipy==1.13.1 gensim==4.3.3 --no-deps
!pip install torch>=2.0.0 torchvision>=0.15.0
!pip install opencv-python
!pip install --upgrade nvidia-cudnn-cu11 nvidia-cublas-cu11
!pip install werkzeug  # Pour TensorBoard

# Configuration (télécharge le repo et prépare l'environnement)
%cd /content
!python colab_setup.py --model-size ad

# Lancer l'entraînement
!python main.py --model-size ad

# Visualiser les résultats dans TensorBoard
%load_ext tensorboard
%tensorboard --logdir /content/yolov5-face/runs/train/face_detection_transfer
```

Le script `main.py` utilise par défaut la version "standard" qui implémente correctement les classes `GatherLayer` et `DistributeLayer` pour une meilleure cohérence avec l'architecture originale décrite dans l'article.

## Exportation du modèle

Pour exporter le modèle entraîné au format ONNX:

```python
# Dans Google Colab, après l'entraînement
!cd /content/yolov5-face && python export.py --weights runs/train/face_detection_transfer/weights/best.pt --include onnx --simplify

# Copier le modèle exporté vers Google Drive
!cp /content/yolov5-face/runs/train/face_detection_transfer/weights/best.onnx /content/drive/MyDrive/YOLOv5_Face_Results/
```

## Intégration avec l'application Flutter

Le modèle ADYOLOv5-Face peut être facilement intégré dans une application Flutter:

1. Copiez le fichier ONNX exporté dans le dossier des assets:

```bash
cp best.onnx /path/to/flutter-face-app/assets/models/adyolov5s.onnx
```

2. Mettez à jour le fichier `pubspec.yaml`:

```yaml
assets:
  - assets/models/adyolov5s.onnx
```

3. Implémentez le code de détection:

```dart
import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';
import 'package:image/image.dart' as imglib;
import 'package:onnxruntime/onnxruntime.dart';

class FaceDetector {
  OrtSession? _session;
  
  Future<void> initialize() async {
    // Charger le modèle ONNX
    final modelData = await rootBundle.load('assets/models/adyolov5s.onnx');
    final buffer = modelData.buffer.asUint8List();
    
    // Créer une session ONNX
    _session = await OrtSession.create(buffer);
  }
  
  Future<List<Rect>> detectFaces(CameraImage image, {double threshold = 0.5}) async {
    if (_session == null) {
      throw Exception("Le détecteur n'est pas initialisé");
    }
    
    // Prétraitement de l'image
    final inputImage = _preprocessImage(image);
    
    // Préparer les entrées du modèle
    final inputs = {
      'images': OrtTensor.fromList(inputImage, [1, 3, 640, 640])
    };
    
    // Exécuter l'inférence
    final outputs = await _session!.run(inputs);
    
    // Posttraitement pour extraire les boîtes de visages
    return _postProcessOutput(outputs, threshold);
  }
  
  List<double> _preprocessImage(CameraImage image) {
    // Conversion et prétraitement de l'image depuis CameraImage
    // ...
    return processedImageData;
  }
  
  List<Rect> _postProcessOutput(Map<String, OrtTensor> outputs, double threshold) {
    // Extraire et traiter les prédictions du modèle
    // ...
    return detectedFaces;
  }
}
```

## Références

- Wang, G., Liu, L., Miao, Q. (2023). "ADYOLOv5-Face: An Enhanced YOLO-Based Face Detector for Small Target Faces." *Electronics*, 13(21), 4184. https://doi.org/10.3390/electronics13214184
- Qi, D., Tan, W., Yao, Q., Liu, J. (2021). "YOLO5Face: Why Reinventing a Face Detector." *arXiv preprint arXiv:2105.12931*.
- Wang, G., Wang, Y., Zhang, H., Gu, R., Hwang, J.-N. (2023). "Gold-YOLO: Efficient Object Detector via Gather-and-Distribute Mechanism." *Neural Information Processing Systems*.
- Jocher, G. et al. (2021). "YOLOv5 by Ultralytics." GitHub repository, https://github.com/ultralytics/yolov5.