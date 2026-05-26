# Gemma-Pruning

Projet de pruning non supervisé pour Gemma, centré sur la compression des couches FFN à partir d'un échantillon de données local. Le dépôt contient deux briques principales : le téléchargement du dataset Hugging Face au format local et le script de pruning qui capture les activations, estime l'information mutuelle entre neurones, puis supprime les neurones redondants avant de sauvegarder un modèle élagué.

## Contenu du projet

- [main.py](main.py) : point d'entrée minimal du dépôt.
- [scripts/download_dataset.py](scripts/download_dataset.py) : télécharge le dataset `ECE-ILAB/resilient-ai-unified` depuis Hugging Face et le sauvegarde localement avec `datasets.save_to_disk`.
- [scripts/prune.py](scripts/prune.py) : charge `google/gemma-4-E4B-it`, capture les activations des couches MLP via hooks, calcule une matrice de distances basée sur l'information mutuelle, applique MDS puis KMeans, et reconstruit les couches linéaires après suppression des neurones redondants.
- [data/](data/) : emplacement attendu pour le dataset local.
- [model/](model/) : dossier de sortie pour les modèles exportés.

## Pipeline

1. Télécharger le dataset localement avec `scripts/download_dataset.py`.
2. Lancer le pruning avec `scripts/prune.py`.
3. Le script charge le tokenizer et le modèle en `bfloat16`, enregistre les activations des hooks sur `down_proj`, sous-échantillonne les tokens pour limiter la mémoire, calcule l'information mutuelle par blocs, puis identifie les neurones à conserver dans chaque cluster.
4. Le modèle élagué est ensuite sauvegardé avec sa configuration Hugging Face mise à jour, y compris `model.config.intermediate_size`.

## Fondations mathématiques

L'algorithme du projet repose sur une estimation non supervisée de la redondance entre neurones FFN à partir d'une information mutuelle définie dans un espace de Hilbert à noyau reproduisant (RKHS).

### Entropie de Rényi d'ordre $\alpha$

Pour un neurone $Z_k$ observé sur $N$ échantillons, on construit une matrice de Gram $K_k$ avec un noyau gaussien RBF :

$$K_k(i, j) = \exp\left(-\frac{(z_{k,i} - z_{k,j})^2}{2\sigma^2}\right)$$

Après normalisation par la trace, la matrice $\tilde{K}_k = \frac{K_k}{\mathrm{tr}(K_k)}$ permet de calculer l'entropie de Rényi :

$$S_\alpha(\tilde{K}_k) = \frac{1}{1-\alpha} \log_2\left(\sum_{i=1}^{N} \lambda_i^\alpha\right)$$

où $\{\lambda_1, \dots, \lambda_N\}$ sont les valeurs propres de $\tilde{K}_k$.

### Information mutuelle entre deux neurones

L'entropie jointe de deux neurones $k$ et $l$ est estimée à l'aide du produit de Hadamard de leurs matrices de noyau : $K_{\text{joint}} = K_k \odot K_l$. L'information mutuelle est alors :

$$I(Z_k ; Z_l) = S_\alpha(\tilde{K}_k) + S_\alpha(\tilde{K}_l) - S_\alpha(\tilde{K}_{\text{joint}})$$

### Largeur de noyau et clustering

La largeur de noyau $\sigma$ suit une règle de Scott empirique dépendant du nombre d'échantillons $N$ et de la dimension $d$ :

$$\sigma = \gamma N^{-\frac{1}{4+d}}$$

Les scores d'information mutuelle sont ensuite transformés en distances, projetés avec MDS, puis regroupés avec KMeans. Dans chaque cluster, seul le neurone le plus proche du centroïde est conservé, les autres étant supprimés physiquement des matrices `gate_proj`, `up_proj` et `down_proj`.

## Installation

L'installation se fait avec `uv` et l'environnement se prépare simplement avec :

```powershell
uv sync
```

L'index PyTorch CUDA 12.4 est déjà déclaré dans [pyproject.toml](pyproject.toml), donc aucune commande d'installation supplémentaire n'est nécessaire.

## Utilisation

Pour récupérer le dataset puis lancer le pruning :

```powershell
uv run python scripts/download_dataset.py
uv run python scripts/prune.py
```

Le script de pruning utilise par défaut :

```python
MODEL_ID = "google/gemma-4-E4B-it"
DATASET_PATH = os.path.join(BASE_DIR, "data", "unified")
SAVE_PATH = os.path.join(BASE_DIR, "models", "gemma4-pruned-mi")
```

Adapte ces chemins si ton environnement local diffère de la structure du dépôt.

## Remarques

- Le script est pensé pour des exécutions gourmandes en VRAM et bénéficie de `device_map="auto"`.
- La partie analytique repose sur `torch`, `transformers`, `datasets`, `scikit-learn`, `numpy` et `tqdm`.
- Le point d'entrée `main.py` est volontairement minimal et peut servir de base pour des tests ou une orchestration future.