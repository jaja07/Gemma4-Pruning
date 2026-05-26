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

# ==========================================
# ÉTAPE 1 : Chargement
# ==========================================
def load_gemma_model(model_id="google/gemma-4-E4B-it"):
    print(f"Chargement du tokenizer pour {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

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
# ÉTAPE 2 : Capture des activations
# ==========================================
def capture_activations(model, tokenizer, dataset_path, num_samples=200):
    """
    Fait passer un échantillon de données dans le modèle et capture les activations FFN.
    """
    print(f"Chargement du dataset depuis {dataset_path}...")
    # On charge le dataset unifié mentionné dans votre README
    dataset = load_from_disk(dataset_path)
    
    # On prend un petit échantillon déterministe (pour la reproductibilité seed=42)
    sample_data = dataset["test"].shuffle(seed=42).select(range(num_samples)) # pyright: ignore
    
    # Dictionnaire pour stocker les activations capturées par couche
    activations = {i: [] for i in range(len(model.model.layers))}
    
    # Définition de la fonction Hook
    def get_activation_hook(layer_idx):
        def hook(module, input, output):
            act = input[0].detach().cpu()
            # Transformation de (batch, seq_len, dim) vers (batch * seq_len, dim)
            act = act.reshape(-1, act.shape[-1])
            activations[layer_idx].append(act)
        return hook

    # On attache les hooks sur la couche 'down_proj' de chaque bloc MLP
    handles = []
    print("Mise en place des hooks sur les couches FFN...")
    for i, layer in enumerate(model.model.layers):
        handle = layer.mlp.down_proj.register_forward_hook(get_activation_hook(i))
        handles.append(handle)

    print("Passage des données dans le modèle (Inférence)...")
    model.eval()
    with torch.no_grad(): # Indispensable pour ne pas stocker les gradients (économise la VRAM)
        for item in tqdm(sample_data):
            # Adaptation basique du texte (à ajuster selon la structure exacte de votre UnifiedSample)
            # Si c'est du multimodal, il faudra passer l'image au processor ici
            text = item.get("text_prompt", item.get("question", "")) # pyright: ignore
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Le forward pass déclenche automatiquement nos hooks
            model(**inputs)

    # Nettoyage : on retire les hooks pour ne pas polluer les futures exécutions
    for handle in handles:
        handle.remove()

    # Concaténation des activations par couche pour faciliter le calcul de l'Information Mutuelle
    for i in activations.keys():
        activations[i] = torch.cat(activations[i], dim=0) # pyright: ignore
        
    print("Capture terminée ! Les activations sont prêtes pour l'analyse.")
    return activations


# ==========================================
# ÉTAPE 3 : Calcul de l'Information Mutuelle
# ==========================================

def compute_kernel_matrix(z, sigma):
    """
    Calcule la matrice de noyau RBF pour un vecteur d'activations d'un neurone.
    z : tenseur 1D de taille (N_samples)
    """
    # z.unsqueeze(1) -> colonne, z.unsqueeze(0) -> ligne
    # La diffusion (broadcasting) crée une matrice des différences au carré
    dist_sq = (z.unsqueeze(1) - z.unsqueeze(0)) ** 2
    
    # Noyau RBF (Radial Basis Function)
    K = torch.exp(-dist_sq / (2 * (sigma ** 2)))
    return K

def compute_renyi_entropy(K, alpha=1.01):
    """
    Calcule l'entropie de Rényi d'ordre alpha pour une matrice de noyau.
    """
    # Normalisation de la matrice pour que la trace soit égale à 1
    trace_K = torch.trace(K)
    if trace_K == 0:
        return torch.tensor(0.0)
    
    K_norm = K / trace_K
    
    # Calcul des valeurs propres (eigvalsh est optimisé pour les matrices symétriques)
    # On ajoute une petite valeur epsilon sur la diagonale pour la stabilité numérique
    epsilon = 1e-8
    K_norm = K_norm + torch.eye(K_norm.size(0), device=K_norm.device) * epsilon
    
    eigvals = torch.linalg.eigvalsh(K_norm)
    
    # On ignore les valeurs propres négatives (artefacts numériques)
    eigvals = torch.clamp(eigvals, min=1e-10)
    
    # Formule de l'entropie de Rényi
    entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eigvals ** alpha))
    return entropy

def calculate_mi_between_neurons(z_k, z_l, sigma, alpha=1.01):
    """
    Calcule l'Information Mutuelle I(Z_k ; Z_l) entre deux neurones.
    """
    # Matrices de noyaux individuelles
    K_k = compute_kernel_matrix(z_k, sigma)
    K_l = compute_kernel_matrix(z_l, sigma)
    
    # Entropies individuelles
    S_k = compute_renyi_entropy(K_k, alpha)
    S_l = compute_renyi_entropy(K_l, alpha)
    
    # Matrice de noyau jointe (Produit de Hadamard / élément par élément)
    K_joint = K_k * K_l
    S_joint = compute_renyi_entropy(K_joint, alpha)
    
    # Information Mutuelle
    MI = S_k + S_l - S_joint
    return max(0.0, MI.item()) # Le MI ne peut pas être négatif théoriquement

def compute_mutual_information(activations, alpha=1.01, gamma=1.0, compression_rate=0.1):
    """
    Identifie les neurones redondants en utilisant le clustering basé sur l'Information Mutuelle
    tel que décrit dans la Section 3.3.3 du papier.
    compression_rate: pourcentage de neurones à élaguer (ex: 0.1 pour 10%)
    """
    print("\n--- Début de l'analyse de l'Information Mutuelle et Clustering ---")
    redundant_pairs_by_layer = {}
    
    for layer_idx, acts in activations.items():
        print(f"Analyse de la couche {layer_idx}...")
        
        N = acts.shape[0]
        d = acts.shape[1] 
        
        # Règle de Scott pour le paramètre du noyau
        sigma = gamma * math.pow(N, -1 / (4 + d))
        
        # Pour éviter un calcul O(d^2) massif, on analyse un sous-ensemble représentatif
        # ou on effectue le calcul par blocs si la VRAM le permet.
        # Ici on prend un échantillon de neurones pour l'exemple (à ajuster selon ta RAM)
        subset_size = min(d, 2000) 
        subset_indices = torch.randperm(d)[:subset_size]
        acts_subset = acts[:, subset_indices]
        
        print(f"  -> Calcul de la matrice de distance MI sur un sous-ensemble de {subset_size} neurones...")
        
        # Matrice de distance basée sur MI: d(Z_k, Z_l) = exp(-I(Z_k, Z_l))
        # Initialisée à 0
        distance_matrix = np.zeros((subset_size, subset_size))
        
        for i in range(subset_size):
            for j in range(i + 1, subset_size):
                z_k = acts_subset[:, i]
                z_l = acts_subset[:, j]
                mi_score = calculate_mi_between_neurons(z_k, z_l, sigma, alpha)
                
                # Équation (1) du papier: A * exp(-I)
                dist = np.exp(-mi_score) 
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
                
        # Multidimensional Scaling (MDS) pour projeter dans un espace de dimension inférieure
        print("  -> Projection MDS...")
        mds = MDS(n_components=10, dissimilarity='precomputed', random_state=42, normalized_stress='auto')
        mds_coords = mds.fit_transform(distance_matrix)
        
        # Clustering: On regroupe les neurones similaires.
        # Le nombre de clusters correspond au nombre de neurones qu'on veut GARDER dans ce sous-ensemble.
        num_clusters = int(subset_size * (1 - compression_rate))
        print(f"  -> Clustering (KMeans) pour trouver les représentants de {num_clusters} clusters...")
        
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(mds_coords)
        
        layer_redundant_pairs = []
        
        # On garde le neurone le plus proche du centroïde, on prune les autres
        for cluster_id in range(num_clusters):
            cluster_indices = np.where(labels == cluster_id)[0]
            if len(cluster_indices) > 1:
                # Calculer la distance au centroïde
                centroid = kmeans.cluster_centers_[cluster_id]
                distances_to_centroid = np.linalg.norm(mds_coords[cluster_indices] - centroid, axis=1)
                
                # Le représentant est celui avec la distance minimale
                representative_idx = cluster_indices[np.argmin(distances_to_centroid)]
                
                # Tous les autres dans le cluster sont considérés comme redondants
                for idx in cluster_indices:
                    if idx != representative_idx:
                        # On mappe les indices locaux du sous-ensemble vers les indices globaux de la couche
                        global_rep = subset_indices[representative_idx].item()
                        global_redundant = subset_indices[idx].item()
                        
                        # Format attendu par la fonction de pruning: (gardé, supprimé, score_virtuel)
                        layer_redundant_pairs.append((global_rep, global_redundant, 1.0))
        
        redundant_pairs_by_layer[layer_idx] = layer_redundant_pairs
        print(f"  -> {len(layer_redundant_pairs)} neurones marqués pour suppression sur cette couche.")
        
    return redundant_pairs_by_layer

# ==========================================
# ÉTAPE 4 : Pruning et Sauvegarde
# ==========================================
def prune_and_save_model(model, tokenizer, redundant_pairs_by_layer, save_path):
    """
    Découpe physiquement les neurones redondants dans les blocs MLP du modèle
    et sauvegarde la nouvelle architecture et les poids.
    """
    print("\n--- Étape 4 : Découpe physique des neurones et Sauvegarde ---")
    
    # On itère sur chaque couche analysée
    for layer_idx, redundant_pairs in redundant_pairs_by_layer.items():
        if not redundant_pairs:
            continue
            
        # 1. Identifier les neurones à supprimer (on supprime le neurone 'l' de chaque paire)
        indices_to_drop = set()
        # Le format attendu de redundant_pairs est (neurone_gardé, neurone_supprimé, score)
        for k, l, mi_score in redundant_pairs:
            indices_to_drop.add(l)
            
        indices_to_drop = list(indices_to_drop)
        if not indices_to_drop:
            continue
            
        # Récupération du bloc MLP cible
        mlp = model.model.layers[layer_idx].mlp
        current_dim = mlp.gate_proj.weight.shape[0] # Dimension intermédiaire actuelle
        
        # 2. Créer le tenseur des indices à conserver
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
        # On récupère la nouvelle taille sur la première couche élaguée
        first_pruned_layer = list(redundant_pairs_by_layer.keys())[0]
        new_intermediate_size = model.model.layers[first_pruned_layer].mlp.gate_proj.out_features
        model.config.intermediate_size = new_intermediate_size
        print(f"\nMise à jour de la configuration : intermediate_size = {new_intermediate_size}")

    print(f"\nSauvegarde du modèle élagué dans {save_path}...")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print("Opération terminée ! Le modèle est prêt pour le benchmark.")

# ==========================================
# EXECUTION PRINCIPALE
# ==========================================
if __name__ == "__main__":
    MODEL_ID = "google/gemma-4-E4B-it"
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATASET_PATH = os.path.join(BASE_DIR, "data", "unified")
    SAVE_PATH = os.path.join(BASE_DIR, "models", "gemma4-pruned-mi")
    
    print("Démarrage du processus de compression...")
    
    # 1. Init
    model, tokenizer = load_gemma_model(MODEL_ID)
    
    # 2. Calibration
    activations = capture_activations(model, tokenizer, DATASET_PATH)
    
    # 3. Analyse
    mi_scores = compute_mutual_information(activations)
    
    # 4. Découpe et sauvegarde
    prune_and_save_model(model, tokenizer, mi_scores, SAVE_PATH)
    
    print("Compression terminée avec succès !")