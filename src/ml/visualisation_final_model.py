import os
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------
# CONFIG
# -----------------------
# ⚠️ adapte le chemin vers ton CSV de métriques
MODEL_VERSION = "v20_dynamic_gpu_safe"
METRICS_CSV = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    f"../data/model/{MODEL_VERSION}/metrics_dynamic_gpu_safe_v2.csv"
))

assert os.path.exists(METRICS_CSV), f"❌ CSV introuvable : {METRICS_CSV}"

# -----------------------
# LOAD CSV
# -----------------------
df = pd.read_csv(METRICS_CSV)
print(f"✅ Fichier chargé : {METRICS_CSV}")
print(df.head())

# -----------------------
# CLEAN / PREP
# -----------------------
# Supprimer les phases vides si besoin
df = df.dropna(subset=["epoch", "phase"])

# Séparer train / val
df_train = df[df["phase"] == "train"]
df_val = df[df["phase"] == "val"]

# -----------------------
# PLOT SETTINGS
# -----------------------
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["legend.fontsize"] = 10

# -----------------------
# 1️⃣ Loss Discriminator / Generator
# -----------------------
plt.figure()
plt.plot(df_train["epoch"], df_train["loss_D"], label="Loss D (train)", alpha=0.6)
plt.plot(df_train["epoch"], df_train["loss_G"], label="Loss G (train)", alpha=0.6)
plt.plot(df_val["epoch"], df_val["loss_D"], "--", label="Loss D (val)")
plt.plot(df_val["epoch"], df_val["loss_G"], "--", label="Loss G (val)")
plt.title("Loss Discriminator / Generator")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------
# 2️⃣ FID & Inception Score
# -----------------------
plt.figure()
plt.plot(df_val["epoch"], df_val["FID"], color="tab:red", label="FID")
plt.ylabel("FID (↓ mieux)", color="tab:red")
plt.xlabel("Epoch")
plt.title("FID & Inception Score (Validation)")

ax2 = plt.gca().twinx()
ax2.plot(df_val["epoch"], df_val["Inception_Score"], color="tab:blue", label="Inception Score")
ax2.set_ylabel("Inception Score (↑ mieux)", color="tab:blue")

lines, labels = plt.gca().get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, loc="best")

plt.tight_layout()
plt.show()

# -----------------------
# 3️⃣ Fake Correct (qualité des fake)
# -----------------------
plt.figure()
plt.plot(df_val["epoch"], df_val["fake_correct"], color="tab:green", label="Fake Correct (val)")
plt.plot(df_train["epoch"], df_train["fake_correct"], "--", color="tab:olive", label="Fake Correct (train)")
plt.title("Taux de Fake Correct (D croit que G est réel)")
plt.xlabel("Epoch")
plt.ylabel("Proportion")
plt.legend()
plt.tight_layout()
plt.show()

# -----------------------
# 4️⃣ Learning Rates (si inclus dans ton CSV)
# -----------------------
if "lr_D" in df.columns and "lr_G" in df.columns:
    plt.figure()
    plt.plot(df_val["epoch"], df_val["lr_D"], label="lr_D", color="tab:orange")
    plt.plot(df_val["epoch"], df_val["lr_G"], label="lr_G", color="tab:purple")
    plt.title("Évolution des Learning Rates (validation)")
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.legend()
    plt.tight_layout()
    plt.show()

print("📊 Visualisation terminée !")
