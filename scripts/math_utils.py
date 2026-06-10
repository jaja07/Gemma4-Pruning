# scripts/math_utils.py
import torch
import math

def compute_kernel_matrix(z: torch.Tensor, sigma: float) -> torch.Tensor:
    """
    Calcule la matrice de noyau RBF pour un vecteur d'activations d'un neurone.
    z : tenseur 1D de taille (N_samples)
    """
    # La diffusion (broadcasting) crée une matrice des différences au carré
    dist_sq = (z.unsqueeze(1) - z.unsqueeze(0)) ** 2
    
    # Noyau RBF (Radial Basis Function)
    K = torch.exp(-dist_sq / (2 * (sigma ** 2)))
    return K

def compute_renyi_entropy(K: torch.Tensor, alpha: float = 1.01) -> torch.Tensor:
    """
    Calcule l'entropie de Rényi d'ordre alpha pour une matrice de noyau.
    """
    trace_K = torch.trace(K)
    if trace_K == 0:
        return torch.tensor(0.0, device=K.device)
    
    K_norm = K / trace_K
    
    # Ajout d'une petite valeur epsilon sur la diagonale pour la stabilité numérique
    epsilon = 1e-8
    K_norm = K_norm + torch.eye(K_norm.size(0), device=K_norm.device) * epsilon
    
    # Calcul des valeurs propres (optimisé pour les matrices symétriques)
    eigvals = torch.linalg.eigvalsh(K_norm)
    
    # On ignore les valeurs propres négatives (artefacts numériques)
    eigvals = torch.clamp(eigvals, min=1e-10)
    
    # Formule de l'entropie de Rényi
    entropy = (1 / (1 - alpha)) * torch.log2(torch.sum(eigvals ** alpha))
    return entropy

def optimize_kernel_width(z: torch.Tensor, sigma_l: float, max_iter: int = 20, lr: float = 0.1) -> float:
    """
    Optimise le paramètre sigma d'un neurone en maximisant le Kernel Alignment Loss
    par rapport au noyau global de la couche (sigma_l).
    """
    device = z.device
    
    # Initialisation du sigma du neurone avec la valeur de Scott globale
    sigma_n = torch.tensor([sigma_l], requires_grad=True, device=device)
    optimizer = torch.optim.Adam([sigma_n], lr=lr)
    
    # Noyau de référence de la couche (constant)
    K_l = compute_kernel_matrix(z, sigma_l).detach()
    norm_K_l = torch.norm(K_l, p='fro')
    
    for _ in range(max_iter):
        optimizer.zero_grad()
        K_n = compute_kernel_matrix(z, sigma_n) # pyright: ignore
        
        # Kernel Alignment = <K_l, K_n>_F / (||K_l||_F * ||K_n||_F)
        inner_product = torch.sum(K_l * K_n)
        norm_K_n = torch.norm(K_n, p='fro')
        
        alignment = inner_product / (norm_K_l * norm_K_n + 1e-8)
        
        # Maximiser l'alignement équivaut à minimiser son opposé
        loss = -alignment
        loss.backward()
        optimizer.step()
        
        # S'assurer que sigma_n reste strictement positif
        with torch.no_grad():
            sigma_n.clamp_(min=1e-3)
            
    return sigma_n.item()

def calculate_mi_between_neurons(z_k: torch.Tensor, z_l: torch.Tensor, sigma_l: float, alpha: float = 1.01) -> float:
    """
    Calcule l'Information Mutuelle I(Z_k ; Z_l) entre deux neurones avec optimisation de sigma.
    """
    # 1. Optimisation des largeurs de noyau individuelles
    sigma_k = optimize_kernel_width(z_k, sigma_l)
    sigma_l_opt = optimize_kernel_width(z_l, sigma_l)
    
    # 2. Matrices de noyaux individuelles optimisées
    K_k = compute_kernel_matrix(z_k, sigma_k)
    K_l = compute_kernel_matrix(z_l, sigma_l_opt)
    
    # 3. Entropies individuelles
    S_k = compute_renyi_entropy(K_k, alpha)
    S_l = compute_renyi_entropy(K_l, alpha)
    
    # 4. Matrice de noyau jointe (Hadamard product)
    K_joint = K_k * K_l
    S_joint = compute_renyi_entropy(K_joint, alpha)
    
    # Information Mutuelle
    MI = S_k + S_l - S_joint
    return max(0.0, MI.item())