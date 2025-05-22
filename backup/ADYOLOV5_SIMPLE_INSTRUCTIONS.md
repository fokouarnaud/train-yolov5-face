# Instructions d'entraînement ADYOLOv5-Face (Version Simple)

Cette version utilise `adyolov5s_simple.yaml` qui implémente le mécanisme Gather-and-Distribute (GD) en utilisant des couches standard de YOLOv5 (`Concat` et `Conv`) au lieu des classes personnalisées `GatherLayer` et `DistributeLayer`. Cette approche évite les erreurs d'importation lors de l'entraînement.

## 1. Commandes d'entraînement sur Google Colab

```python
# ADYOLOv5-Face: Entraînement et Évaluation avec la version simple
# Utilise le repo modifié avec le mécanisme Gather-Distribute implémenté sans classes personnalisées

# Étape 1: Monter Google Drive et copier les scripts
from google.colab import drive
drive.mount('/content/drive')

# Copier les scripts depuis Drive (incluant le nouveau script ADYOLOv5-Face)
!mkdir -p /content
!cp /content/drive/MyDrive/yolov5_face_scripts/main.py \
   /content/drive/MyDrive/yolov5_face_scripts/data_preparation.py \
   /content/drive/MyDrive/yolov5_face_scripts/model_training.py \
   /content/drive/MyDrive/yolov5_face_scripts/model_evaluation.py \
   /content/drive/MyDrive/yolov5_face_scripts/utils.py \
   /content/drive/MyDrive/yolov5_face_scripts/colab_setup.py \
   /content/drive/MyDrive/yolov5_face_scripts/config.py \
   /content/drive/MyDrive/yolov5_face_scripts/adyolov5_training.py \
   /content/drive/MyDrive/yolov5_face_scripts/export_adyolov5_onnx_for_flutter.py \
   /content/

# Étape 2: Installer les dépendances compatibles
!pip install numpy==1.26.4 scipy==1.13.1 gensim==4.3.3 --no-deps
!pip install torch>=2.0.0 torchvision>=0.15.0
!pip install opencv-python
!pip install --upgrade nvidia-cudnn-cu11 nvidia-cublas-cu11
!pip install werkzeug  # Pour TensorBoard
!pip install onnx onnxruntime onnx-simplifier  # Pour l'exportation ONNX

# Étape 3: Utiliser le script adyolov5_training.py avec l'option "simple"
%cd /content
!python adyolov5_training.py --batch-size 32 --epochs 300 --model-type simple

# Étape 4: Visualiser les résultats
%load_ext tensorboard
%tensorboard --logdir /content/yolov5-face/runs/train/face_detection_transfer

# Étape 5: Exporter le modèle en ONNX pour l'application Flutter
!python export_adyolov5_onnx_for_flutter.py \
   --weights /content/yolov5-face/runs/train/face_detection_transfer/weights/best.pt \
   --img-size 320 \
   --simplify \
   --dynamic-batch \
   --half \
   --flutter-path "/content/drive/MyDrive/YOLOv5_Face_Results/flutter_app_path"

# Étape 6: Copier le modèle exporté vers Google Drive
!mkdir -p /content/drive/MyDrive/YOLOv5_Face_Results/ADYOLOv5_Face
!cp /content/export/adyolov5_face_320_simple_half_quantized.onnx /content/drive/MyDrive/YOLOv5_Face_Results/ADYOLOv5_Face/
```

## 2. Comment utiliser le modèle ADYOLOv5-Face dans l'application Flutter

1. Ajoutez le chemin du modèle dans `model_file.dart`:
   ```dart
   static const String adYoloV5Face320x320DynamicBatchonnx =
       'assets/models/adyolov5_face_320_simple_half_quantized.onnx';
   ```

2. Assurez-vous que le fichier est inclus dans `pubspec.yaml`:
   ```yaml
   flutter:
     assets:
       - assets/models/adyolov5_face_320_simple_half_quantized.onnx
   ```

3. Utilisation dans votre code:
   ```dart
   // Pour utiliser le modèle standard YOLOv5-Face
   final detector = YoloOnnxFaceDetection.instance;
   
   // Pour utiliser ADYOLOv5-Face pour une meilleure détection des petits visages
   final adDetector = YoloOnnxFaceDetection.adyolo();
   
   await adDetector.init();
   final faces = await adDetector.predict(imageData);
   ```

## Notes importantes

1. Cette version utilise une implémentation alternative du mécanisme Gather-and-Distribute qui évite les problèmes d'importation des classes personnalisées.

2. La version simple (`adyolov5s_simple.yaml`) devrait avoir des performances similaires à la version standard, mais avec une meilleure compatibilité.

3. Les tailles d'entrée recommandées:
   - 320x320: Pour les appareils mobiles (bon équilibre performance/précision)
   - 640x640: Pour une précision maximale sur les petits visages
