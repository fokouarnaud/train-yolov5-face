#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script simplifié pour corriger directement le fichier loss.py sans dépendances complexes"""

import os
import sys

def fix_loss_py_file():
    """
    Corrige le problème de conversion de type dans loss.py en utilisant un remplacement direct
    """
    print("\n" + "=" * 70)
    print(" CORRECTION SIMPLIFIÉE DU FICHIER LOSS.PY ".center(70, "="))
    print("=" * 70 + "\n")
    
    # Chemin absolu vers le fichier (pour éviter les problèmes liés au répertoire de travail)
    loss_file = '/content/yolov5-face/utils/loss.py'
    
    # Vér