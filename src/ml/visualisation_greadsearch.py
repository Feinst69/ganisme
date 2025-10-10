import os
import pandas as pd
import matplotlib.pyplot as plt

# === CONFIG ===
report_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/reporting/fine_tuning_report_split.csv"))
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data/reporting/figures_all_metrics"))
os.makedirs(output_dir, exist_ok=True)

# === CHARGEMENT ===
df = pd.read_csv(report_csv)
print(f"✅ Données chargées ({len(df)} lignes)")

# === LISTE DES MÉTRIQUES À PLOTER ===
metrics = [
    "loss_D", "loss_G",
    "precision_train", "recall_train", "f1_train", "acc_train",
    "precision_val", "recall_val", "f1_val", "acc_val"
]

# === FONCTION DE PLOT ===
def plot_metric(metric_name, df, output_dir):
    plt.figure(figsize=(8, 5))
    for act, data in df.groupby("activation_D"):
        grouped = data.groupby("epoch")[metric_name].mean()
        plt.plot(grouped.index, grouped.values, label=act)
    
    plt.title(f"{metric_name} par epoch (moyenne par activation_D)")
    plt.xlabel("Époque")
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{metric_name}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()
    return save_path

# === GÉNÉRATION DE TOUS LES GRAPHIQUES ===
saved_paths = []
for m in metrics:
    path = plot_metric(m, df, output_dir)
    saved_paths.append(path)
    print(f"📈 Figure sauvegardée : {os.path.basename(path)}")

# === FIGURE MULTI-MÉTRIQUES (train vs val) ===
train_metrics = ["precision_train", "recall_train", "f1_train", "acc_train"]
val_metrics = ["precision_val", "recall_val", "f1_val", "acc_val"]

plt.figure(figsize=(10, 6))
for metric_train, metric_val in zip(train_metrics, val_metrics):
    plt.plot(df.groupby("epoch")[metric_train].mean(), "--", label=f"{metric_train}")
    plt.plot(df.groupby("epoch")[metric_val].mean(), "-", label=f"{metric_val}")
plt.title("Comparaison moyenne des métriques train vs validation")
plt.xlabel("Époque")
plt.ylabel("Valeur moyenne")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
multi_path = os.path.join(output_dir, "train_vs_val_comparison.png")
plt.savefig(multi_path, dpi=200)
plt.close()
print(f"📊 Figure globale sauvegardée : {os.path.basename(multi_path)}")

# === RAPPORT SYNTHÉTIQUE (uniquement les epochs complètes à 50) ===
df_last = df[df["epoch"] == 50]  # ⬅️ ne garder que les modèles ayant atteint l'époque 50

if df_last.empty:
    print("\n Aucun modèle n'a encore atteint 50 époques. Impossible d'afficher un top 5.")
else:
    best = df_last.sort_values("f1_val", ascending=False).head(5)

    print("\n🏆 Top 5 versions (epoch = 50) selon F1 Validation :")
    print(best[[
        "version", "activation_D", "lr_G", "lr_D", "dropout_D", 
        "label_noise", "f1_val", "acc_val"
    ]])

print(f"\n📁 Toutes les figures sont enregistrées dans : {output_dir}")
