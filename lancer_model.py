import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from train import Arguments, train, train_m_models
from plotter import plot_loss_accs
from checkpointing import get_extrema_performance_steps_per_trials

# Set device globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# Liste des valeurs de rtrain à tester
rtrain_values = np.arange(0.1, 1.0, 0.1)

# Dictionnaire pour stocker les résultats
results = {}

# Fonction pour entraîner le modèle et récupérer les métriques
def train_and_evaluate(model_name, r_train):
    args = Arguments()

    # Configuration des données
    args.p = 31
    args.operator = "+"
    args.r_train = r_train  # On modifie rtrain ici
    args.operation_orders = 2
    args.train_batch_size = 512
    args.eval_batch_size = 2**12
    args.num_workers = 0

    # Configuration du modèle
    args.model = model_name
    args.num_heads = 4 if model_name == "gpt" else None
    args.num_layers = 2
    args.embedding_size = 2**7
    args.hidden_size = 2**7 if model_name == "lstm" else None
    args.dropout = 0.0
    args.share_embeddings = False
    args.bias_classifier = True

    # Optimisation
    args.optimizer = 'adamw'
    args.lr = 1e-3
    args.weight_decay = 1e-3

    # Entraînement
    args.n_steps = 100
    args.eval_first = 1
    args.eval_period = 1
    args.print_step = 1
    args.save_model_step = 10
    args.save_statistic_step = 10

    # Expérimentation
    args.device = device  # Use the global device
    args.exp_id = 0
    args.exp_name = f"experiment_{model_name}_r{r_train}"
    args.log_dir = '../logs'
    args.verbose = False  # Réduire la sortie console

    print(f"📌 Training {model_name} with r_train={r_train} on {device}")
    

    # Entraîner le modèle
    all_models_per_trials, all_metrics, all_checkpoint_paths = train_m_models(args, M=2, seeds=[0, 42])

    # Stocker les performances extrêmes
    extrema_performances = get_extrema_performance_steps_per_trials(all_metrics)

    return extrema_performances

# Boucle sur les valeurs de r_train
for r in rtrain_values:
    print(f"🔄 Entraînement LSTM avec r_train={r}...")

    try:
      lstm_results = train_and_evaluate("lstm", r)
    except Exception as e:
      print(f"Erreur lors de l'entraînement LSTM avec r_train={r}: {e}")

    # lstm_results = train_and_evaluate("lstm", r)

    print(f"🔄 Entraînement GPT avec r_train={r}...")
    gpt_results = train_and_evaluate("gpt", r)

    # Stocker les résultats
    results[r] = {"lstm": lstm_results, "gpt": gpt_results}

# Déterminer les valeurs limites de rtrain
min_rtrain, max_rtrain = None, None

for r in rtrain_values:
    # Convertir sur CPU puis en numpy avant de calculer la moyenne
    aval_mean = np.mean(results[r]["lstm"]["Aval"]["mean"].cpu().numpy())
    if aval_mean >= 0.9:
        min_rtrain = r
        break

for r in rtrain_values[::-1]:
    atrain_mean = np.mean(results[r]["lstm"]["Atrain"]["mean"].cpu().numpy())
    aval_mean = np.mean(results[r]["lstm"]["Aval"]["mean"].cpu().numpy())
    if atrain_mean >= 0.99 and aval_mean <= 0.5:
        max_rtrain = r
        break

print(f"✅ Plus petite valeur de rtrain pour Aval >= 0.9 : {min_rtrain}")
print(f"✅ Plus grande valeur de rtrain pour Atrain ≈ 1.0 et Aval ≤ 0.5 : {max_rtrain}")

# Tracer les courbes de performance en fonction de r_train
fig, ax1 = plt.subplots(figsize=(10, 5))

r_vals = list(results.keys())
lstm_avals = [np.mean(results[r]["lstm"]["Aval"]["mean"].cpu().numpy()) for r in r_vals]
gpt_avals = [np.mean(results[r]["gpt"]["Aval"]["mean"].cpu().numpy()) for r in r_vals]

ax1.plot(r_vals, lstm_avals, label="LSTM Aval", marker="o")
ax1.plot(r_vals, gpt_avals, label="GPT Aval", marker="s")
ax1.set_xlabel("r_train")
ax1.set_ylabel("Aval")
ax1.legend()
ax1.grid()
ax1.set_title("Précision de validation en fonction de r_train")

plt.show()
