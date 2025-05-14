#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script pour nettoyer les fichiers de poids corrompus ou vides
"""

import os
import argparse
import torch
import logging
from config import DEFAULT_PATHS

def setup_logging():
    """Configure le logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def parse_args():
    """Parse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(
        description='Script pour nettoyer les fichiers de poids corrompus ou vides'
    )
    parser.add_argument('--weights-dir', type=str, default=DEFAULT_PATHS["weights_dir"],
                        help='Répertoire contenant les fichiers de poids')
    parser.add_argument('--check-integrity', action='store_true',
                        help='Vérifier l\'intégrité des fichiers de poids PyTorch')
    parser.add_argument('--delete-empty', action='store_true', default=True,
                        help='Supprimer les fichiers vides')
    parser.add_argument('--delete-corrupt', action='store_true',
                        help='Supprimer les fichiers corrompus (peut supprimer des fichiers valides mais non-PyTorch)')
    
    return parser.parse_args()

def list_weight_files(weights_dir):
    """Liste tous les fichiers de poids dans le répertoire"""
    return [f for f in os.listdir(weights_dir) if f.endswith('.pt')]

def check_empty_files(weights_dir, delete=False):
    """Vérifie les fichiers vides et les supprime si demandé"""
    empty_files = []
    
    for filename in list_weight_files(weights_dir):
        filepath = os.path.join(weights_dir, filename)
        if os.path.getsize(filepath) == 0:
            empty_files.append(filename)
            if delete:
                try:
                    os.remove(filepath)
                    logging.info(f"✓ Fichier vide supprimé: {filename}")
                except OSError as e:
                    logging.error(f"✗ Erreur lors de la suppression de {filename}: {e}")
    
    if not empty_files:
        logging.info("✓ Aucun fichier vide trouvé")
    elif not delete:
        logging.warning(f"⚠️ Fichiers vides trouvés: {', '.join(empty_files)}")
    
    return empty_files

def check_corrupt_files(weights_dir, delete=False):
    """Vérifie les fichiers corrompus et les supprime si demandé"""
    corrupt_files = []
    
    for filename in list_weight_files(weights_dir):
        filepath = os.path.join(weights_dir, filename)
        
        # Ignorer les fichiers vides (déjà gérés par check_empty_files)
        if os.path.getsize(filepath) == 0:
            continue
        
        try:
            # Tenter de charger le fichier avec PyTorch
            torch.load(filepath, map_location='cpu', weights_only=False)
        except Exception as e:
            corrupt_files.append(filename)
            logging.warning(f"⚠️ Fichier corrompu trouvé: {filename} (Erreur: {e})")
            
            if delete:
                try:
                    os.remove(filepath)
                    logging.info(f"✓ Fichier corrompu supprimé: {filename}")
                except OSError as e:
                    logging.error(f"✗ Erreur lors de la suppression de {filename}: {e}")
    
    if not corrupt_files:
        logging.info("✓ Aucun fichier corrompu trouvé")
    elif not delete:
        logging.warning(f"⚠️ Fichiers corrompus trouvés: {', '.join(corrupt_files)}")
    
    return corrupt_files

def main():
    """Point d'entrée principal du script"""
    setup_logging()
    args = parse_args()
    
    # Vérifier si le répertoire des poids existe
    if not os.path.exists(args.weights_dir):
        logging.error(f"✗ Le répertoire {args.weights_dir} n'existe pas")
        return False
    
    logging.info(f"=== Vérification des fichiers de poids dans {args.weights_dir} ===")
    
    # Nombre total de fichiers de poids
    weight_files = list_weight_files(args.weights_dir)
    logging.info(f"Nombre total de fichiers de poids: {len(weight_files)}")
    
    # Vérifier les fichiers vides
    empty_files = check_empty_files(args.weights_dir, delete=args.delete_empty)
    
    # Vérifier l'intégrité des fichiers si demandé
    if args.check_integrity:
        corrupt_files = check_corrupt_files(args.weights_dir, delete=args.delete_corrupt)
    
    # Résumé
    logging.info("=== Résumé du nettoyage ===")
    logging.info(f"Fichiers vides trouvés: {len(empty_files)}")
    if args.check_integrity:
        logging.info(f"Fichiers corrompus trouvés: {len(corrupt_files)}")
    
    if (len(empty_files) == 0) and (not args.check_integrity or len(corrupt_files) == 0):
        logging.info("✓ Tous les fichiers sont valides!")
        return True
    else:
        if args.delete_empty or (args.check_integrity and args.delete_corrupt):
            logging.info("✓ Nettoyage terminé!")
        else:
            logging.warning("⚠️ Des problèmes ont été détectés mais aucun fichier n'a été supprimé")
            logging.info("Pour supprimer les fichiers problématiques, utilisez --delete-empty et/ou --delete-corrupt")
        return False

if __name__ == "__main__":
    main()
