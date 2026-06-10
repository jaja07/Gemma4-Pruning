import os
from datasets import load_dataset, Dataset

# def download_calibration_data(dataset_id: str, save_dir: str, token: str = None): #pyright: ignore
#     """
#     Télécharge uniquement le split 'calibration' d'un dataset Hugging Face
#     et le sauvegarde localement au format Arrow.
#     """
#     print(f"Téléchargement du split 'calibration' de '{dataset_id}' depuis Hugging Face...")
    
#     # 1. Téléchargement uniquement du split "calibration" (en utilisant les arguments de la fonction)
#     ds_calib = load_dataset(
#         dataset_id, 
#         split="calibration", 
#         trust_remote_code=True, 
#         token=token
#     )
    
#     print("\nDataset de calibration téléchargé avec succès. Aperçu :")
#     print(ds_calib)
    
#     # 2. Création du dossier cible basé sur l'argument 'save_dir'
#     print(f"\nSauvegarde locale dans {save_dir}...")
#     os.makedirs(save_dir, exist_ok=True)
    
#     # 3. Sauvegarde au format Arrow
#     ds_calib.save_to_disk(save_dir) # pyright: ignore
    
#     print("\nSauvegarde terminée ! Les données de calibration sont prêtes.")

def download_calibration_data(dataset_id: str, save_dir: str, token: str = None): # pyright: ignore
    """
    Extrait à la volée 200 échantillons de calibration sans télécharger 
    les dizaines de gigaoctets du dataset complet grâce au streaming.
    """
    print(f"Connexion au dataset '{dataset_id}' en mode STREAMING...")
    
    # 1. Chargement en flux (streaming=True : téléchargement instantané de 0 Go)
    ds_stream = load_dataset(
        dataset_id, 
        split="calibration", 
        streaming=True, 
        trust_remote_code=True, 
        token=token
    )
    
    # 2. Extraction stricte des 200 échantillons requis depuis le flux internet
    print("Extraction rapide des 200 premiers échantillons de calibration...")
    samples = list(ds_stream.take(200)) # pyright: ignore
    
    # 3. Conversion instantanée en un Dataset Arrow local
    ds_local = Dataset.from_list(samples) # pyright: ignore
    
    # 4. Sauvegarde locale de ce mini-dataset (quelques Mo à peine)
    print(f"Sauvegarde locale de l'échantillon extrait dans {save_dir}...")
    os.makedirs(save_dir, exist_ok=True)
    ds_local.save_to_disk(save_dir)
    
    print("\n[Succès] Vos 200 exemples de calibration sont prêts localement en quelques secondes !")