import os
import torch
import torch.nn as nn
import math
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import numpy as np
from sklearn.manifold import MDS
from sklearn.cluster import KMeans

from scripts.math_utils import calculate_mi_between_neurons

def get_mlp_modules(model):
    """
    Parcourt dynamiquement l'ensemble du modèle et extrait tous les blocs MLP,
    quelle que soit l'architecture (VLM, CausalLM, ConditionalGeneration).
    """
    mlps = []
    for name, module in model.named_modules():
        # On identifie un bloc MLP valide s'il possède les projections attendues
        if name.endswith("mlp") and hasattr(module, "down_proj") and hasattr(module, "gate_proj"):
            mlps.append(module)
            
    if not mlps:
        raise ValueError("Aucun bloc MLP trouvé dans l'architecture du modèle !")
        
    return mlps

# ==========================================
def load_gemma_model(model_id="google/gemma-4-E4B-it", token : str = None): #pyright: ignore
    print(f"Chargement du tokenizer pour {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)

    print(f"Chargement du modèle {model_id} en bfloat16...")
    # L'utilisation de device_map="auto" permet de répartir le modèle 
    # automatiquement sur le GPU disponible.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    return model, tokenizer

# ==========================================
def capture_activations(model, tokenizer, dataset_path, num_samples=200, max_tokens=2000):
    """
    Fait passer un échantillon de données dans le modèle et capture les activations FFN.
    Intègre un sous-échantillonnage pour éviter la saturation mémoire en O(N^2).
    """
    print(f"Chargement du dataset depuis {dataset_path}...")
    dataset = load_from_disk(dataset_path)
    sample_data = dataset.shuffle(seed=42).select(range(num_samples)) # pyright: ignore
    
    # --- LA CORRECTION EST ICI ---
    mlps = get_mlp_modules(model)
    activations = {i: [] for i in range(len(mlps))}
    # -----------------------------
    
    def get_activation_hook(layer_idx):
        def hook(module, input, output):
            act = input[0].detach().cpu()
            act = act.reshape(-1, act.shape[-1])
            activations[layer_idx].append(act)
        return hook

    handles = []
    print("Mise en place des hooks sur les couches FFN...")
    # On itère directement sur la liste dynamique des MLPs
    for i, mlp in enumerate(mlps):
        handle = mlp.down_proj.register_forward_hook(get_activation_hook(i))
        handles.append(handle)

    print("Passage des données dans le modèle (Inférence)...")
    model.eval()
    with torch.no_grad(): # Indispensable pour ne pas stocker les gradients (économise la VRAM)
        for item in tqdm(sample_data):
            # Adaptation basique du texte (à ajuster selon la structure exacte de votre UnifiedSample)
            text = item.get("text_prompt", item.get("question", "")) # pyright: ignore
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Le forward pass déclenche automatiquement nos hooks
            model(**inputs)

    # Nettoyage : on retire les hooks pour ne pas polluer les futures exécutions
    for handle in handles:
        handle.remove()

    print(f"Sous-échantillonnage des activations à {max_tokens} tokens maximum par couche...")
    # Concaténation et sous-échantillonnage des activations par couche
    for i in activations.keys():
        acts = torch.cat(activations[i], dim=0) # pyright: ignore
        
        # Sélection aléatoire si le nombre de tokens dépasse la limite fixée
        if acts.shape[0] > max_tokens:
            indices = torch.randperm(acts.shape[0])[:max_tokens]
            acts = acts[indices]
            
        activations[i] = acts # type: ignore
        
    print("Capture terminée ! Les activations sont prêtes pour l'analyse.")
    return activations

# ==========================================
def compute_mutual_information(activations, alpha=1.01, gamma=1.0, compression_rate=0.1, block_size=2048):
    """
    Identifie les neurones redondants par blocs pour traiter toute la couche.
    """
    print("\n--- Début de l'analyse de l'Information Mutuelle et Clustering ---")
    redundant_pairs_by_layer = {}
    
    for layer_idx, acts in activations.items():
        print(f"Analyse de la couche {layer_idx}...")
        
        N = acts.shape[0]
        d = acts.shape[1] 
        
        # Règle de Scott globale pour la couche (sigma_l)
        sigma_l = gamma * math.pow(N, -1 / (4 + d))
        
        layer_redundant_pairs = []
        
        # Découpage de la couche en blocs pour traiter TOUS les neurones sans O(d^2) massif
        num_blocks = math.ceil(d / block_size)
        
        for block_idx in range(num_blocks):
            start_idx = block_idx * block_size
            end_idx = min(start_idx + block_size, d)
            current_block_size = end_idx - start_idx
            
            print(f"  -> Traitement du bloc {block_idx+1}/{num_blocks} ({current_block_size} neurones)...")
            acts_block = acts[:, start_idx:end_idx]
            
            distance_matrix = np.zeros((current_block_size, current_block_size))
            
            # Calcul de la matrice de distance MI pour ce bloc
            for i in range(current_block_size):
                for j in range(i + 1, current_block_size):
                    z_k = acts_block[:, i]
                    z_l = acts_block[:, j]
                    mi_score = calculate_mi_between_neurons(z_k, z_l, sigma_l, alpha)
                    
                    dist = np.exp(-mi_score) 
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist

            # Num_clusters correspond au nombre de neurones à GARDER dans ce bloc
            num_clusters = int(current_block_size * (1 - compression_rate))
            
            # Gérer le cas où le bloc est trop petit
            if num_clusters < 1 or num_clusters >= current_block_size:
                continue

            # MDS Projection
            # NOTE : Le papier recommande de tester M graines aléatoires et de garder celle
            # minimisant la divergence KL. Pour des raisons de performance d'exécution,
            # nous fixons ici une graine par défaut, mais ce paramètre peut être itéré.
            mds = MDS(n_components=10, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
            mds_coords = mds.fit_transform(distance_matrix)
            
            # Clustering
            kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
            labels = kmeans.fit_predict(mds_coords)
            
            # Extraction des représentants
            for cluster_id in range(num_clusters):
                cluster_indices = np.where(labels == cluster_id)[0]
                if len(cluster_indices) > 1:
                    centroid = kmeans.cluster_centers_[cluster_id]
                    distances_to_centroid = np.linalg.norm(mds_coords[cluster_indices] - centroid, axis=1)
                    
                    representative_local_idx = cluster_indices[np.argmin(distances_to_centroid)]
                    
                    for local_idx in cluster_indices:
                        if local_idx != representative_local_idx:
                            global_rep = start_idx + representative_local_idx
                            global_redundant = start_idx + local_idx
                            layer_redundant_pairs.append((global_rep, global_redundant, 1.0))
                            
        redundant_pairs_by_layer[layer_idx] = layer_redundant_pairs
        print(f"  -> Total : {len(layer_redundant_pairs)} neurones marqués pour suppression sur cette couche.")
        
    return redundant_pairs_by_layer

# ==========================================
def prune_and_save_model(model, tokenizer, redundant_pairs_by_layer, save_path):
    """
    Découpe physiquement les neurones redondants dans les blocs MLP du modèle
    et sauvegarde la nouvelle architecture et les poids.
    """
    print("\n--- Étape 4 : Découpe physique des neurones et Sauvegarde ---")
    
    mlps = get_mlp_modules(model)
    # -----------------------------
    
    for layer_idx, redundant_pairs in redundant_pairs_by_layer.items():
        if not redundant_pairs:
            continue
            
        indices_to_drop = set()
        for k, l, mi_score in redundant_pairs:
            indices_to_drop.add(l)
            
        indices_to_drop = list(indices_to_drop)
        if not indices_to_drop:
            continue
            
        # On utilise le bloc MLP récupéré dynamiquement
        mlp = mlps[layer_idx]
        current_dim = mlp.gate_proj.weight.shape[0]
        
        indices_to_keep = [i for i in range(current_dim) if i not in indices_to_drop]
        keep_tensor = torch.tensor(indices_to_keep, device=model.device)
        
        print(f"Couche {layer_idx} : Suppression de {len(indices_to_drop)} neurones. Reste : {len(indices_to_keep)}")
        
        # 3. Découpe de gate_proj (on réduit les out_features, donc la dimension 0)
        old_gate = mlp.gate_proj
        new_gate = nn.Linear(old_gate.in_features, len(indices_to_keep), bias=old_gate.bias is not None)
        new_gate.weight.data = torch.index_select(old_gate.weight.data, 0, keep_tensor)
        new_gate = new_gate.to(model.device).to(model.dtype)
        mlp.gate_proj = new_gate
        
        # 4. Découpe de up_proj (on réduit les out_features, dimension 0)
        old_up = mlp.up_proj
        new_up = nn.Linear(old_up.in_features, len(indices_to_keep), bias=old_up.bias is not None)
        new_up.weight.data = torch.index_select(old_up.weight.data, 0, keep_tensor)
        new_up = new_up.to(model.device).to(model.dtype)
        mlp.up_proj = new_up
        
        # 5. Découpe de down_proj (on réduit les in_features, dimension 1)
        old_down = mlp.down_proj
        new_down = nn.Linear(len(indices_to_keep), old_down.out_features, bias=old_down.bias is not None)
        new_down.weight.data = torch.index_select(old_down.weight.data, 1, keep_tensor)
        # Le bias de down_proj (s'il y en a un) n'est pas affecté car l'output de la couche reste le même
        if old_down.bias is not None:
            new_down.bias.data = old_down.bias.data 
        new_down = new_down.to(model.device).to(model.dtype)
        mlp.down_proj = new_down

    # === Mise à jour de la configuration globale ===
    # Dans HuggingFace, on modifie la config globale pour refléter la nouvelle taille du MLP.
    # On présuppose ici un élagage uniforme (même nombre de neurones supprimés sur chaque couche modifiée).
    if redundant_pairs_by_layer:
            first_pruned_layer = list(redundant_pairs_by_layer.keys())[0]
            # On lit la nouvelle taille directement sur le module modifié
            new_intermediate_size = mlps[first_pruned_layer].gate_proj.out_features
            model.config.intermediate_size = new_intermediate_size
            print(f"\nMise à jour de la configuration globale : intermediate_size = {new_intermediate_size}")

    print(f"\nSauvegarde du modèle élagué dans {save_path}...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Opération terminée ! Le modèle est prêt pour le benchmark.")