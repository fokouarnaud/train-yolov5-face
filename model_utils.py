#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utilitaires pour gérer les modèles et les problèmes de compatibilité
"""

import torch
import numpy as np
import os
import sys

def safe_load_model(model_path, map_location='cpu', force_weights_only=None):
    """Charge un modèle PyTorch de manière sécurisée en gérant les changements de PyTorch 2.6+
    
    Args:
        model_path (str): Chemin vers le modèle à charger
        map_location (str): Appareil sur lequel charger le modèle
        force_weights_only (bool, optional): Force weights_only=True/False. Si None, utilise l'approche la plus sûre
    
    Returns:
        model: Le modèle chargé
    """
    print(f"Chargement du modèle: {model_path}")
    
    if force_weights_only is not None:
        # Si on force l'utilisation d'une option particulière
        try:
            return torch.load(model_path, map_location=map_location, weights_only=force_weights_only)
        except Exception as e:
            print(f"Erreur lors du chargement avec weights_only={force_weights_only}: {e}")
            raise
    
    # Approche 1: Essayer avec la méthode la plus sûre (context manager)
    try:
        if hasattr(torch.serialization, 'safe_globals'):
            print("Utilisation de safe_globals pour charger le modèle")
            with torch.serialization.safe_globals(['numpy.core.multiarray._reconstruct']):
                return torch.load(model_path, map_location=map_location)
        else:
            # Versions de PyTorch qui n'ont pas encore safe_globals
            raise AttributeError("safe_globals n'est pas disponible")
    except Exception as e1:
        print(f"Méthode 1 échouée: {e1}")
        
        # Approche 2: Essayer avec weights_only=False (moins sûre mais plus compatible)
        try:
            print("Tentative avec weights_only=False")
            return torch.load(model_path, map_location=map_location, weights_only=False)
        except Exception as e2:
            print(f"Méthode 2 échouée: {e2}")
            
            # Approche 3: Essayer la méthode standard (pour les anciennes versions de PyTorch)
            try:
                print("Tentative avec la méthode standard")
                return torch.load(model_path, map_location=map_location)
            except Exception as e3:
                print(f"Méthode 3 échouée: {e3}")
                
                # Si tout échoue, lever l'erreur avec un message clair
                print("ERREUR: Impossible de charger le modèle!")
                raise RuntimeError(f"Échec du chargement du modèle {model_path}. Erreurs: {e1}, puis {e2}, puis {e3}")
