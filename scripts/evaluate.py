import os
import time
import torch
import numpy as np
import math
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk

def compute_model_size_and_flops_ratio(model_orig, model_pruned):
    """
    Calcule la taille des modèles et estime le ratio théorique de FLOPs
    en se concentrant sur la réduction des couches FFN.
    """
    params_orig = sum(p.numel() for p in model_orig.parameters())
    params_pruned = sum(p.numel() for p in model_pruned.parameters())
    
    # Estimation des FLOPs basée sur la taille intermédiaire du MLP
    # Dans Gemma/LLaMA, un bloc MLP a 3 projections linéaires: gate_proj, up_proj, down_proj
    hidden_size = model_orig.config.hidden_size
    num_layers = model_orig.config.num_hidden_layers
    
    inter_orig = model_orig.config.intermediate_size
    inter_pruned = model_pruned.config.intermediate_size
    
    # FLOPs FFN totaux approximatifs (2 * in * out par opération linéaire)
    flops_ffn_orig = 2 * inter_orig * hidden_size * 3 * num_layers
    flops_ffn_pruned = 2 * inter_pruned * hidden_size * 3 * num_layers
    
    relative_flops_ffn = (flops_ffn_pruned / flops_ffn_orig) * 100
    
    print("\n=== Métriques de Structure et Taille ===")
    print(f"Modèle Original - intermediate_size: {inter_orig} | Paramètres: {params_orig:,}")
    print(f"Modèle Élagué   - intermediate_size: {inter_pruned} | Paramètres: {params_pruned:,}")
    print(f"Réduction brute des paramètres: {((params_orig - params_pruned) / params_orig) * 100:.2f}%")
    print(f"Relative FLOPs (FFN uniquement) : {relative_flops_ffn:.2f}%")
    
    return relative_flops_ffn

def evaluate_generation_speed(model, tokenizer, prompts, max_new_tokens=50):
    """
    Mesure la latence d'inférence et le débit en tokens par seconde.
    """
    model.eval()
    total_tokens = 0
    total_time = 0.0
    
    # Warmup pour stabiliser les mesures du GPU
    inputs = tokenizer("Warmup prompt", return_tensors="pt").to(model.device)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=10)
        
    print(f"\nMesure de la vitesse d'inférence sur {len(prompts)} prompts...")
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        input_len = inputs["input_ids"].shape[1]
        
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        end_time = time.time()
        
        gen_len = outputs.shape[1] - input_len
        total_tokens += gen_len
        total_time += (end_time - start_time)
        
    speed = total_tokens / total_time if total_time > 0 else 0
    return speed, total_time

def compute_kl_and_perplexity(model_orig, model_pruned, tokenizer, dataset_path, num_samples=50):
    """
    Calcule la Perplexité (PPL) du modèle élagué et la divergence KL des logits
    par rapport au modèle original pour évaluer la dégradation de la qualité linguistique.
    """
    print("\nChargement du dataset de test pour l'évaluation de la fidélité sémantique...")
    dataset = load_from_disk(dataset_path)
    samples = dataset["test"].shuffle(seed=42).select(range(num_samples))
    
    model_orig.eval()
    model_pruned.eval()
    
    total_kl = 0.0
    total_loss_pruned = 0.0
    total_tokens = 0
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    
    print("Calcul de la divergence KL et de la perplexité...")
    with torch.no_grad():
        for item in tqdm(samples):
            text = item.get("text_prompt", item.get("question", ""))
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model_orig.device) for k, v in inputs.items()}
            
            labels = inputs["input_ids"].clone()
            
            # Forward pass original
            outputs_orig = model_orig(**inputs)
            logits_orig = outputs_orig.logits
            
            # Forward pass élagué
            outputs_pruned = model_pruned(**inputs)
            logits_pruned = outputs_pruned.logits
            
            # Aligner les séquences temporelles pour exclure le dernier token prédit
            shift_logits_orig = logits_orig[..., :-1, :].contiguous()
            shift_logits_pruned = logits_pruned[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # 1. Calcul de la Perplexité (Cross Entropy) pour le modèle élagué
            loss_pruned = loss_fct(shift_logits_pruned.view(-1, shift_logits_pruned.size(-1)), shift_labels.view(-1))
            total_loss_pruned += loss_pruned.sum().item()
            total_tokens += shift_labels.numel()
            
            # 2. Calcul de la Divergence KL entre les distributions de tokens (Origine || Pruned)
            p = torch.nn.functional.softmax(shift_logits_orig, dim=-1)
            log_p = torch.nn.functional.log_softmax(shift_logits_orig, dim=-1)
            log_q = torch.nn.functional.log_softmax(shift_logits_pruned, dim=-1)
            
            kl_div = torch.sum(p * (log_p - log_q), dim=-1)
            total_kl += kl_div.mean().item()
            
    avg_loss = total_loss_pruned / total_tokens if total_tokens > 0 else 0
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    avg_kl = total_kl / num_samples
    
    print(f"\n=== Métriques de Fidélité Sémantique ===")
    print(f"Perplexité du modèle élagué : {perplexity:.4f}")
    print(f"Divergence KL moyenne des logits : {avg_kl:.6f}")
    
    return perplexity, avg_kl

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_ORIG_ID = "google/gemma-4-E4B-it"
    MODEL_PRUNED_PATH = os.path.join(BASE_DIR, "models", "gemma4-pruned-mi")
    DATASET_PATH = os.path.join(BASE_DIR, "data", "unified")
    
    print("Démarrage du script d'évaluation comparative...")
    
    # 1. Chargement du Tokenizer partagé
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ORIG_ID)
    
    # 2. Chargement des modèles
    print(f"Chargement du modèle original : {MODEL_ORIG_ID}")
    model_orig = AutoModelForCausalLM.from_pretrained(MODEL_ORIG_ID, device_map="auto", torch_dtype=torch.bfloat16)
    
    print(f"Chargement du modèle élagué depuis : {MODEL_PRUNED_PATH}")
    if not os.path.exists(MODEL_PRUNED_PATH):
        print(f"Erreur : Le modèle élagué n'existe pas dans '{MODEL_PRUNED_PATH}'. Veuillez d'abord exécuter prune.py.")
        exit(1)
    model_pruned = AutoModelForCausalLM.from_pretrained(MODEL_PRUNED_PATH, device_map="auto", torch_dtype=torch.bfloat16)
    
    # 3. Évaluation structurelle et théorique des FLOPs
    compute_model_size_and_flops_ratio(model_orig, model_pruned)
    
    # 4. Évaluation de la vitesse d'inférence
    test_prompts = [
        "Explique-moi les fondements de la relativité générale en quelques lignes.",
        "Écris une fonction Python récursive pour calculer le nième terme de Fibonacci.",
        "Quelles sont les principales différences entre Docker et une machine virtuelle ?",
        "Résume les avantages de l'architecture Multi-Agent pour l'automatisation financière."
    ]
    
    speed_orig, time_orig = evaluate_generation_speed(model_orig, tokenizer, test_prompts)
    speed_pruned, time_pruned = evaluate_generation_speed(model_pruned, tokenizer, test_prompts)
    
    print(f"\n=== Métriques de Performance Matérielle ===")
    print(f"Débit du modèle original : {speed_orig:.2f} tokens/seconde (Temps total: {time_orig:.2f}s)")
    print(f"Débit du modèle élagué   : {speed_pruned:.2f} tokens/seconde (Temps total: {time_pruned:.2f}s)")
    if speed_orig > 0:
        print(f"Gain de vitesse d'inférence constaté : +{((speed_pruned - speed_orig) / speed_orig) * 100:.2f}%")
        
    # 5. Évaluation de la fidélité sémantique (KL & Perplexité)
    try:
        compute_kl_and_perplexity(model_orig, model_pruned, tokenizer, DATASET_PATH, num_samples=30)
    except Exception as e:
        print(f"\n[Avertissement] Impossible de calculer la Perplexité/KL : {e}")
        print("Assurez-vous que le dataset de calibration au format Arrow est bien présent dans votre dossier local.")
