# ADYOLOv5-Face: Détecteur de visages amélioré avec mécanisme GD

Ce projet intègre ADYOLOv5-Face au framework YOLOv5-Face existant. ADYOLOv5-Face est une version améliorée qui utilise un mécanisme Gather-and-Distribute (GD) et une tête de détection supplémentaire pour améliorer la détection des petits visages.

## Sommaire

- [Présentation](#présentation)
- [Différences avec YOLOv5s-Face](#différences-avec-yolov5s-face)
- [Performances](#performances)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Entraînement sur Google Colab](#entraînement-sur-google-colab)
- [Exportation du modèle](#exportation-du-modèle)
- [Intégration avec l'application Flutter](#intégration-avec-lapplication-flutter)
- [Références](#références)

## Présentation

ADYOLOv5-Face est une version améliorée de YOLOv5s-Face spécifiquement conçue pour améliorer la détection des petits visages dans des environnements complexes. Ce modèle introduit deux améliorations principales:

1. **Mécanisme Gather-and-Distribute (GD)**: Remplace la structure Feature Pyramid Network (FPN) + Path Aggregation Network (PAN) traditionnelle pour améliorer la fusion des caractéristiques entre les couches non adjacentes, réduisant ainsi la perte d'informations.

2. **Tête de détection supplémentaire**: Ajoute une quatrième tête de détection spécifique pour les petits visages, permettant une meilleure détection des visages de petite taille.

## Différences avec YOLOv5s-Face

### Architecture générale

YOLOv5s-Face utilise une architecture standard composée de trois parties principales:
- **Backbone**: CSPDarknet53 pour l'extraction des caractéristiques
- **Neck**: FPN+PAN pour la fusion des caractéristiques multi-échelles
- **Head**: 3 têtes de détection pour les visages de différentes tailles

ADYOLOv5-Face conserve le même backbone mais modifie les deux autres composants:
- **Neck**: Mécanisme GD qui améliore la fusion des caractéristiques entre les couches non adjacentes
- **Head**: 4 têtes de détection incluant une tête supplémentaire spécifique pour les petits visages

### Mécanisme Gather-and-Distribute

Le mécanisme GD se compose de deux phases:

1. **Phase de rassemblement (Gather)**: Collecte et fusionne les informations de toutes les couches
2. **Phase de distribution (Distribute)**: Redistribue les informations fusionnées à chaque couche

Ce mécanisme permet une meilleure préservation des informations lors de la fusion des caractéristiques de différentes échelles, ce qui est particulièrement important pour la détection des petits objets.

### Tête de détection supplémentaire

ADYOLOv5-Face ajoute une quatrième tête de détection avec une résolution plus élevée (160×160) pour améliorer la détection des petits visages. Cette tête utilise des informations plus détaillées des couches peu profondes, ce qui permet une meilleure détection des visages de taille 4×4 pixels.

## Performances

ADYOLOv5-Face montre des améliorations significatives par rapport à YOLOv5s-Face sur le jeu de données WiderFace:

| Modèle | Easy (%) | Medium (%) | Hard (%) |
|--------|----------|------------|----------|
| YOLOv5s-Face | 94,33 | 92,61 | 83,15 |
| ADYOLOv5-Face | 94,80 | 93,77 | 84,37 |
| **Amélioration** | **+0,47** | **+1,16** | **+1,22** |

Les améliorations les plus significatives concernent les sous-ensembles "Medium" et "Hard", qui contiennent généralement des visages plus petits, partiellement occultés ou dans des conditions d'éclairage difficiles.

## Installation

### Prérequis

- Python 3.8 ou supérieur
- PyTorch 2.0 ou supérieur
- CUDA (recommandé pour l'entraînement)

### Configuration locale

1. Cloner le dépôt YOLOv5-Face:

```bash
git clone https://github.com/fokouarnaud/yolov5-face.git
cd yolov5-face
```

2. Installer les dépendances:

```bash
pip install -r requirements.txt
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

# Charger le modèle
model = attempt_load('weights/adyolov5s.pt', map_location='cuda:0')
model.eval()

# Préparer l'image
img = cv2.imread('test.jpg')
img0 = img.copy()
img = letterbox(img, 640)[0]
img = img[:, :, ::-1].transpose(2, 0, 1)
img = np.ascontiguousarray(img)
img = torch.from_numpy(img).float().div(255.0).unsqueeze(0).to('cuda:0')

# Inférence
pred = model(img)[0]
pred = non_max_suppression(pred, 0.5, 0.5)

# Traiter les détections
for i, det in enumerate(pred):
    if len(det):
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
        for *xyxy, conf, cls in det:
            # Dessiner la boîte
            x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
            cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Afficher le score
            score = f"{conf:.2f}"
            cv2.putText(img0, score, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Afficher le résultat
cv2.imshow('Result', img0)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Entraînement sur Google Colab

Pour entraîner ADYOLOv5-Face sur Google Colab, utilisez le script `adyolov5_training.py`:

1. Téléchargez les scripts nécessaires sur Google Drive:
   - `adyolov5_training.py`
   - Autres scripts du dossier `reconnaissance_Facial`

2. Exécutez dans Google Colab:

```python
# Monter Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copier et exécuter le script d'entraînement
!cp /content/drive/MyDrive/yolov5_face_scripts/adyolov5_training.py /content/
!python adyolov5_training.py --batch-size 32 --epochs 300
```

## Exportation du modèle

Pour exporter le modèle entraîné au format ONNX pour une utilisation dans l'application Flutter:

```python
# Dans Google Colab, après l'entraînement
!python /content/yolov5-face/export.py --weights /content/yolov5-face/runs/train/face_detection_transfer/weights/best.pt --include onnx

# Copier le modèle exporté vers Google Drive
!cp /content/yolov5-face/runs/train/face_detection_transfer/weights/best.onnx /content/drive/MyDrive/YOLOv5_Face_Results/
```

## Intégration avec l'application Flutter

Pour utiliser le modèle ADYOLOv5-Face dans l'application Flutter:

1. Copiez le fichier ONNX exporté dans le dossier des assets de l'application Flutter:

```bash
cp best.onnx /path/to/flutter-face-app/assets/models/
```

2. Mettez à jour le fichier `pubspec.yaml` pour inclure le nouveau modèle:

```yaml
assets:
  - assets/models/adyolov5s.onnx
```

3. Mettez à jour le code pour utiliser le nouveau modèle:

```dart
// Initialiser le modèle
final modelPath = 'assets/models/adyolov5s.onnx';
final faceDetector = await loadModel(modelPath);

// Fonction de chargement du modèle
Future<FaceDetector> loadModel(String modelPath) async {
  final modelData = await rootBundle.load(modelPath);
  final modelBytes = modelData.buffer.asUint8List();
  
  final options = OrtSessionOptions();
  final session = OrtSession.create(modelBytes, options);
  
  return FaceDetector(session);
}

// Classe de détection de visages
class FaceDetector {
  final OrtSession _session;
  
  FaceDetector(this._session);
  
  Future<List<Face>> detect(Uint8List imageBytes, {double threshold = 0.5}) async {
    // Code de prétraitement et d'inférence
    // ...
  }
}
```

## Références

- Wang, G., Liu, L., Miao, Q. (2023). "ADYOLOv5-Face: An Enhanced YOLO-Based Face Detector for Small Target Faces." *Electronics*, 13(21), 4184.
- Qi, D., Tan, W., Yao, Q., Liu, J. (2021). "YOLO5Face: Why Reinventing a Face Detector." *arXiv preprint arXiv:2105.12931*.
- Wang, G., Wang, Y., Zhang, H., Gu, R., Hwang, J.-N. (2023). "Gold-YOLO: Efficient Object Detector via Gather-and-Distribute Mechanism." *Neural Information Processing Systems*.
