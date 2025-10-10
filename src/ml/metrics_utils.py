# metrics_utils.py
import torch
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights
from torchmetrics.image.fid import FrechetInceptionDistance

def setup_inception_and_fid(device):
    inception = inception_v3(weights=Inception_V3_Weights.DEFAULT).to(device)
    inception.eval()
    if hasattr(inception, 'AuxLogits'):
        inception.AuxLogits = None
    fid_metric = FrechetInceptionDistance(feature=2048).to(device)
    mean = torch.tensor([0.485,0.456,0.406],device=device).view(1,3,1,1)
    std  = torch.tensor([0.229,0.224,0.225],device=device).view(1,3,1,1)
    return inception, fid_metric, mean, std

@torch.no_grad()
def compute_inception_score(inception_model, imgs, mean, std):
    imgs = F.interpolate(imgs, size=(299,299), mode='bilinear', align_corners=False)
    imgs = (imgs.clamp(-1,1)+1)/2
    imgs = (imgs - mean) / std
    logits = inception_model(imgs)
    probs = F.softmax(logits, dim=1)
    p_y = probs.mean(dim=0, keepdim=True)
    kl = probs * (torch.log(probs + 1e-8) - torch.log(p_y + 1e-8))
    return torch.exp(kl.sum(dim=1).mean()).item()

def compute_reward(fid, IS, fake_correct):
    """Plus le FID est bas, mieux c’est. IS et fake_correct élevés = bon."""
    reward = -(fid / 300) + 0.5 * (IS / 5) + 0.5 * fake_correct
    return reward
