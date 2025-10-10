# dynamic_utils.py
import numpy as np

def apply_rl_action(action, optimizer_G, optimizer_D, n_critic, g_steps, min_lr, max_lr):
    lrG_mult, lrD_mult, d_adj, g_adj = action
    for g in optimizer_G.param_groups:
        g['lr'] = float(np.clip(g['lr'] * (1 + 0.1 * lrG_mult), min_lr, max_lr))
    for g in optimizer_D.param_groups:
        g['lr'] = float(np.clip(g['lr'] * (1 + 0.1 * lrD_mult), min_lr, max_lr))
    n_critic = int(np.clip(n_critic + round(d_adj), 1, 10))
    g_steps = int(np.clip(g_steps + round(g_adj), 1, 5))
    return optimizer_G, optimizer_D, n_critic, g_steps
