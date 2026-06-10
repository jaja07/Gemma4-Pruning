import os
import argparse
import logging

# Import de ta configuration (idéalement via Pydantic Settings comme évoqué précédemment)
from config import settings 

# Imports de tes modules actuels
from scripts.download_dataset import download_calibration_data
from scripts.prune import (
    load_gemma_model, 
    capture_activations, 
    compute_mutual_information,  # type: ignore
    prune_and_save_model
)

# Configuration basique du logger pour le terminal
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_pipeline(force_download=False):
    """
    Exécute le pipeline complet : Téléchargement -> Capture -> Analyse MI -> Pruning -> Export
    """
    print(f"DEBUG TOKEN -> Présent: {bool(settings.hf_token)} | Longueur: {len(settings.hf_token)} | Commence par: {settings.hf_token[:7] if settings.hf_token else 'Aucun'}")
    # 1. Définition des chemins (centralisés)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data", "unified")
    SAVE_MODEL_DIR = os.path.join(BASE_DIR, "model", "gemma4-pruned-mi")
    
    MODEL_ID = "google/gemma-4-E4B-it"
    DATASET_ID = "ECE-ILAB/resilient-ai-unified"

    logger.info("=== DÉMARRAGE DU PIPELINE DE COMPRESSION GEMMA 4 ===")

    # 2. Téléchargement du dataset (Split calibration)
    if not os.path.exists(DATA_DIR) or force_download:
        logger.info(f"Étape 1/4 : Téléchargement des données de calibration depuis {DATASET_ID}...")
        download_calibration_data(DATASET_ID, DATA_DIR, token=settings.hf_token)
    else:
        logger.info("Étape 1/4 : Données de calibration déjà présentes. Ignoré.")

    # 3. Chargement du modèle
    logger.info(f"Étape 2/4 : Chargement du modèle {MODEL_ID}...")
    model, tokenizer = load_gemma_model(MODEL_ID, token=settings.hf_token)

    # 4. Capture des activations
    logger.info("Étape 3/4 : Capture des activations (Inférence sur le split de calibration)...")
    activations = capture_activations(model, tokenizer, DATA_DIR, num_samples=200)

    # 5. Calcul de l'Information Mutuelle
    logger.info("Étape 4/4 : Calcul de l'Information Mutuelle et identification des neurones redondants...")
    mi_scores = compute_mutual_information(activations)

    # 6. Découpe et Export
    logger.info(f"Étape Finale : Découpe physique et sauvegarde dans {SAVE_MODEL_DIR}...")
    os.makedirs(SAVE_MODEL_DIR, exist_ok=True)
    prune_and_save_model(model, tokenizer, mi_scores, SAVE_MODEL_DIR)

    logger.info("=== PIPELINE TERMINÉ AVEC SUCCÈS ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de pruning basé sur l'Information Mutuelle")
    parser.add_argument("--force-download", action="store_true", help="Force le re-téléchargement du dataset")
    args = parser.parse_args()

    run_pipeline(force_download=args.force_download)