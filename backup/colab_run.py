# YOLOv5-Face: Entraînement et Évaluation
# Version pour PyTorch 2.6+ et Python 3.11

# Étape 1: Monter Google Drive et copier les scripts
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

# Étape 2: Installer les dépendances compatibles
!pip install numpy==1.26.4 scipy==1.13.1 gensim==4.3.3 --no-deps
!pip install torch>=2.0.0 torchvision>=0.15.0
!pip install opencv-python
# Optimisations CUDA pour A100
!pip install --upgrade nvidia-cudnn-cu11 nvidia-cublas-cu11
# Installer werkzeug pour résoudre le problème de TensorBoard
!pip install werkzeug

# Étape 3: Exécuter le script de configuration
%cd /content
!python colab_setup.py --model-size s

# Rappel des modifications manuelles à faire sur le repo local
print("\n===== RAPPEL DES MODIFICATIONS MANUELLES =====")
print("Pour que l'entraînement et l'évaluation fonctionnent correctement avec Python 3.11 et PyTorch 2.6+, vous devez avoir modifié manuellement les fichiers suivants dans votre repo local :")

print("\n1. Compatibilité NumPy 1.26+ :")
print("   - box_overlaps.pyx: Remplacer np.int par np.int64 et np.int_t par np.int64_t")
print("   - utils/face_datasets.py: Remplacer .astype(np.int) par .astype(np.int32)")
print("   - Tous les fichiers: Remplacer np.float par np.float64")

print("\n2. Compatibilité PyTorch 2.6+ :")
print("   - test_widerface.py: Remplacer:")
print("     pred = model(img, augment=opt.augment)[0]")
print("     Par:")
print("     outputs = model(img, augment=opt.augment)")
print("     pred = outputs[0] if isinstance(outputs, tuple) else outputs")

print("\n3. Éviter les divisions par zéro dans evaluation.py:")
print("   - Modifier la fonction dataset_pr_info pour inclure:")
print("     if pr_curve[i, 0] == 0:")
print("         _pr_curve[i, 0] = 0")
print("     else:")
print("         _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]")
print("     if count_face == 0:")
print("         _pr_curve[i, 1] = 0")
print("     else:")
print("         _pr_curve[i, 1] = pr_curve[i, 1] / count_face")

print("\nSi vous n'avez pas fait ces modifications dans votre repo local, l'entraînement pourrait échouer")
print("ou l'évaluation pourrait donner des AP de 0.0.")
print("==============================================\n")

# Étape 4: Lancer l'entraînement et l'évaluation
!python main.py --model-size s

# Étape 5: Visualiser les résultats
%load_ext tensorboard
%tensorboard --logdir /content/yolov5-face/runs/train/face_detection_transfer
