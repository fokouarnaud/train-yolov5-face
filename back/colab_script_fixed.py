# Étape 1: Monter Google Drive et copier les scripts
from google.colab import drive
drive.mount('/content/drive')

!mkdir -p /content
!cp /content/drive/MyDrive/yolov5_face_scripts/main.py /content/drive/MyDrive/yolov5_face_scripts/data_preparation.py /content/drive/MyDrive/yolov5_face_scripts/model_training.py /content/drive/MyDrive/yolov5_face_scripts/model_evaluation.py /content/drive/MyDrive/yolov5_face_scripts/utils.py /content/drive/MyDrive/yolov5_face_scripts/colab_setup.py /content/drive/MyDrive/yolov5_face_scripts/fix_train_comma.py /content/

# Étape 2: Installer les dépendances compatibles
!pip install numpy==1.26.4 scipy==1.13.1 gensim==4.3.3 --no-deps
!pip install torch>=2.0.0 torchvision>=0.15.0
!pip install opencv-python  # S'assurer que OpenCV est installé

# Étape 3: Exécuter le script de configuration
%cd /content
!python colab_setup.py --model-size s

# Étape 3.5: Corriger spécifiquement le problème de la virgule dans train.py
!python fix_train_comma.py

# Installer werkzeug pour résoudre le problème de TensorBoard
!pip install werkzeug

# Étape 4: Lancer l'entraînement
!python main.py

# Étape 5: Visualiser les résultats
%load_ext tensorboard
%tensorboard --logdir /content/yolov5-face/runs/train/face_detection_transfer

# Étape 6: Créer un dossier pour les résultats
!mkdir -p /content/drive/MyDrive/YOLOv5_Face_Results

# Copier les résultats de l'entraînement vers Google Drive
!cp -r /content/yolov5-face/runs/train/face_detection_transfer /content/drive/MyDrive/YOLOv5_Face_Results/