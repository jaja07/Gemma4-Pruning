import os
from datasets import load_dataset

def main():
    # Le chemin local où ton script de pruning attend les données
    # Ajuste-le si tu es sur Windows et non dans un conteneur WSL
    # ex: DATA_DIR = "C:/Projets/data/unified"
    DATA_DIR = "/workspace/code/data/unified" 
    DATASET_ID = "ECE-ILAB/resilient-ai-unified"

    print(f"Téléchargement du dataset {DATASET_ID} depuis Hugging Face...")
    print("Cela peut prendre du temps (env. 57 Go au total)...")
    
    # 1. Téléchargement. 
    # Mettre trust_remote_code=True est parfois nécessaire pour les datasets persos
    ds = load_dataset(DATASET_ID, trust_remote_code=True)
    
    print("\nDataset téléchargé avec succès. Aperçu des splits :")
    print(ds)
    
    print(f"\nSauvegarde locale dans {DATA_DIR}...")
    # Crée le dossier s'il n'existe pas
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 2. Sauvegarde au format Arrow (le format optimisé attendu par load_from_disk)
    ds.save_to_disk(DATA_DIR)
    
    print("\nSauvegarde terminée ! Tu peux maintenant lancer ton script de pruning.")

if __name__ == "__main__":
    main()