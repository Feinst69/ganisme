# controller_rl.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ------------------------------
# Réseau de contrôle (policy network)
# ------------------------------
class RLController(nn.Module):
    def __init__(self, state_dim=7, hidden_dim=64, action_dim=4):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.sigma = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        sigma = torch.clamp(self.sigma(x), -3, 1)  # bornage
        return mu, sigma.exp()  # retourne moyenne et variance positive


# ------------------------------
# Agent RL (policy gradient)
# ------------------------------
class RLAgent:
    def __init__(self, state_dim=7, action_dim=4, lr=1e-4, gamma=0.99, device="cpu"):
        self.controller = RLController(state_dim, 64, action_dim).to(device)
        self.optimizer = optim.Adam(self.controller.parameters(), lr=lr)
        self.gamma = gamma
        self.device = device

        # Historique des apprentissages
        self.episode_rewards = []
        self.episode_losses = []
        self.episode_actions = []
        self.episode_log_probs = []

    # ------------------------------
    # Sélection d’action (échantillonnage gaussien)
    # ------------------------------
    def select_action(self, state):
        mu, sigma = self.controller(state)
        dist = torch.distributions.Normal(mu, sigma)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()

        # Stockage pour suivi
        self.episode_actions.append(action.detach().cpu().numpy().tolist())
        self.episode_log_probs.append(log_prob.item())

        return action, log_prob

    # ------------------------------
    # Mise à jour de la policy (Policy Gradient)
    # ------------------------------
    def update(self, log_prob, reward):
        loss = -log_prob * reward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Stockage pour logs
        self.episode_rewards.append(reward)
        self.episode_losses.append(loss.item())

        return loss.item()

    # ------------------------------
    # Logs lisibles en console
    # ------------------------------
    def log_epoch(self, epoch, action, reward, loss_rl, optimizer_G, optimizer_D, n_critic, g_steps):
        lrG = optimizer_G.param_groups[0]['lr']
        lrD = optimizer_D.param_groups[0]['lr']
        print(f"🧠 [RL] Epoch {epoch+1:03d} | "
              f"Action={', '.join([f'{a:.3f}' for a in action.tolist()])} | "
              f"Reward={reward:.4f} | Loss={loss_rl:.6f} | "
              f"lr_G={lrG:.6f}, lr_D={lrD:.6f} | "
              f"n_critic={n_critic}, g_steps={g_steps}")

    # ------------------------------
    # Résumé global après l’entraînement
    # ------------------------------
    def summary(self):
        if len(self.episode_rewards) == 0:
            print("⚠️ Aucune donnée de RL enregistrée.")
            return

        avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
        avg_loss = sum(self.episode_losses) / len(self.episode_losses)

        print("\n================ RL TRAINING SUMMARY ================")
        print(f"📈 Moyenne des récompenses : {avg_reward:.4f}")
        print(f"📉 Moyenne des pertes RL    : {avg_loss:.6f}")
        print(f"🎯 Dernières actions        : {self.episode_actions[-5:]}")
        print("=====================================================")
