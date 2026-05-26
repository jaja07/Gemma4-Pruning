# Gemma-Pruning-MI : Élagage Structurel Non Supervisé des LLM basé sur l'Information Mutuelle

Ce projet fournit une implémentation complète et optimisée de la méthode d'élagage (*pruning*) présentée dans le papier de recherche **Large Language Model Pruning** (Huang et al., 2024). Cette approche permet de compresser les grands modèles de langage, spécifiquement calibrée ici pour l'architecture **Gemma 4 (google/gemma-4-E4B-it)**, de manière **totalement non supervisée** et **sans nécessiter de réentraînement**.

L'objectif principal est d'éliminer la redondance au sein des couches de réseaux de neurones d'extension (Feed-Forward Networks, FFN) en utilisant l'estimateur d'entropie de Rényi d'ordre $\alpha$ appliqué à des matrices de noyau, combiné à une stratégie de clustering par réduction de dimensionnalité (MDS + K-Means).

---

## 1. Architecture du Pipeline

Le processus de compression est découpé en quatre étapes majeures, orchestrées de manière séquentielle pour minimiser l'empreinte VRAM et maximiser la précision de l'identification des neurones redondants :

```text
[ Étape 1 : Chargement ]  ──> Charge le modèle Gemma en bfloat16 et son tokenizer
           │
           ▼
[ Étape 2 : Capture ]     ──> Enregistre les activations FFN via hooks (aplatissement token-level)
           │
           ▼
[ Étape 3 : Analyse ]     ──> Calcule l'Information Mutuelle, applique MDS et segmente via KMeans
           │
           ▼
[ Étape 4 : Découpe ]     ──> Suppression physique des poids, ajustement de la config et sauvegarde
```

1. **Chargement et initialisation** : le modèle cible est instancié en précision `bfloat16` avec une distribution automatique sur les ressources matérielles disponibles via `device_map="auto"`.
2. **Capture des activations** : des *PyTorch Forward Hooks* sont greffés sur la sous-couche `down_proj` de chaque bloc FFN. Un échantillon de texte passe dans le modèle. Pour éviter toute distorsion liée au padding et capturer la sémantique fine, les tenseurs 3D `(batch, seq_len, dim)` sont immédiatement projetés et aplatis en matrices 2D `(N_tokens, D_neurons)`. Chaque token est traité comme un échantillon statistique indépendant.
3. **Analyse informationnelle et clustering** : au lieu d'une analyse combinatoire exhaustive en $O(d^2)$ impraticable sur les grands modèles, le script extrait un sous-ensemble représentatif de neurones cachés, évalue leurs profils d'Information Mutuelle, convertit ces scores en distances géométriques, projette l'espace via MDS (*Multidimensional Scaling*) et regroupe les neurones similaires via KMeans.
4. **Découpe physique et sauvegarde** : pour chaque cluster, seul le neurone le plus proche du centroïde est conservé. Les autres sont physiquement supprimés en modifiant les dimensions des matrices de poids des opérateurs linéaires (`gate_proj`, `up_proj`, `down_proj`). La configuration Hugging Face (`model.config.intermediate_size`) est mise à jour pour garantir la parfaite rechargeabilité du modèle.

---

## 2. Description de l'Algorithme Implémenté

### Fondations Mathématiques

L'algorithme repose sur la théorie de l'information appliquée aux espaces de Hilbert à noyau reproduisant (RKHS), permettant d'estimer l'Information Mutuelle sans avoir à modéliser explicitement des distributions de probabilité de haute dimension.

#### 1. Entropie de Rényi d'ordre $\alpha$ matricielle

Pour une variable aléatoire $Z_k$ correspondant aux activations d'un neurone $k$ sur $N$ échantillons, on calcule une matrice de Gram $K_k \in \mathbb{R}^{N \times N}$ via un noyau gaussien RBF (*Radial Basis Function*) :

$$K_k(i, j) = \exp\left(-\frac{(z_{k,i} - z_{k,j})^2}{2\sigma^2}\right)$$

La matrice est normalisée par sa trace pour que $\tilde{K}_k = \frac{K_k}{\text{tr}(K_k)}$. L'entropie matricielle de Rényi d'ordre $\alpha$ est définie par le spectre de valeurs propres $\{\lambda_1, \dots, \lambda_N\}$ de $\tilde{K}_k$ :

$$S_\alpha(\tilde{K}_k) = \frac{1}{1-\alpha} \log_2\left(\sum_{i=1}^{N} \lambda_i^\alpha\right)$$

#### 2. Information Mutuelle entre deux neurones

L'entropie jointe de deux neurones $k$ et $l$ est estimée à l'aide du produit de Hadamard de leurs matrices de noyau respectives : $K_{\text{joint}} = K_k \odot K_l$. L'Information Mutuelle est alors :

$$I(Z_k ; Z_l) = S_\alpha(\tilde{K}_k) + S_\alpha(\tilde{K}_l) - S_\alpha(\tilde{K}_{\text{joint}})$$

#### 3. Règle de Scott empirique pour la largeur de noyau

Le paramètre de bande passante $\sigma$ est déterminant. Conformément à la section 3.4 du papier, il est calculé au niveau de la couche globale en fonction du nombre d'échantillons $N$ et de la dimensionnalité $d$ des neurones cachés :

$$\sigma = \gamma N^{-\frac{1}{4+d}}$$

### Stratégie de mise à l'échelle : clustering géométrique

Pour contourner la complexité quadratique liée à la comparaison de toutes les paires de neurones, le pipeline applique la méthodologie de la section 3.3.3 :

1. Les scores d'Information Mutuelle sont convertis en métrique de dissimilarité : $d(\mathcal{Z}_k, \mathcal{Z}_l) = \exp(-I(Z_k; Z_l))$.
2. L'algorithme **MDS** projette cette matrice de distance dans un espace euclidien restreint, par exemple 10 dimensions.
3. L'algorithme **K-Means** segmente cet espace. Le nombre de clusters cibles est directement dicté par le taux de compression souhaité.
4. Au sein de chaque groupe, le neurone le plus proche du centroïde géométrique est désigné comme le représentant sémantique. Tous les autres neurones du même groupe sont marqués pour l'élagage.

---

## 3. Installation et Configuration de l'Environnement avec `uv`

Ce projet utilise `uv`, un gestionnaire de paquets et d'environnements Python extrêmement rapide écrit en Rust.

### Configuration pour Windows avec support natif CUDA

Sous Windows, pour s'assurer que les dépendances PyTorch précompilées intègrent correctement le support matériel GPU de votre carte NVIDIA, il convient de cibler explicitement l'index binaire approprié, avec CUDA 12.4 de préférence pour une compatibilité maximale.

Exécutez la suite de commandes suivante dans votre terminal PowerShell :

```powershell
# 1. Initialiser le projet et créer la structure de base
uv init gemma-pruning-mi
cd gemma-pruning-mi

# 2. Configurer explicitement l'index PyTorch pour récupérer les versions Windows compatibles CUDA 12.4
uv add torch --index https://download.pytorch.org/whl/cu124

# 3. Ajouter l'écosystème Hugging Face et les outils de Data Science requis
uv add transformers datasets accelerate tqdm scikit-learn numpy
```

### Vérification de la disponibilité de CUDA

Pour confirmer que l'environnement virtuel créé par `uv` accède de manière transparente à votre GPU NVIDIA local, lancez la commande de diagnostic suivante :

```powershell
uv run python -c "import torch; print(f'CUDA disponible : {torch.cuda.is_available()}'); print(f'GPU détecté : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"Aucun\"}')"
```

---

## 4. Guide d'Utilisation

### Configuration des Chemins

Ouvrez le fichier `script.py` et modifiez les constantes au sein du bloc principal `if __name__ == "__main__":` pour refléter vos répertoires locaux :

```python
MODEL_ID = "google/gemma-4-E4B-it"             # Identifiant Hugging Face ou chemin local du modèle
DATASET_PATH = "C:/chemin/vers/votre/dataset"  # Chemin vers le dossier du dataset au format Arrow/Hugging Face
SAVE_PATH = "./models/gemma4-pruned-mi"        # Répertoire de destination du modèle compressé
```

### Exécution du Pipeline

Pour lancer le script complet au sein de l'environnement isolé géré par `uv`, exécutez simplement :

```powershell
uv run script.py
```

Le script affichera la progression pas à pas dans la console, détaillera le volume de neurones supprimés par couche FFN, mettra à jour les configurations sous-jacentes et exportera le modèle prêt à être évalué sur vos bancs de test.