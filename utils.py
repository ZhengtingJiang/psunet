import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.normal_(m.weight, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)
        truncated_normal_(m.bias, mean=0, std=0.001)

def init_weights_orthogonal_normal(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0, std=0.001)
        #nn.init.normal_(m.bias, std=0.001)

def l2_regularisation(m):
    l2_reg = None

    for W in m.parameters():
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    return l2_reg

def save_mask_prediction_example(mask, pred, iter):
    plt.imshow(pred[0,:,:],cmap='Greys')
    plt.savefig('images/'+str(iter)+"_prediction.png")
    plt.imshow(mask[0,:,:],cmap='Greys')
    plt.savefig('images/'+str(iter)+"_mask.png")


import matplotlib.pyplot as plt
import wandb

def visualize_patch_center_slice(inputs, targets, preds, step, tag="Train"):
    """
    可视化 patch 的中间一层（z 轴切片）
    - inputs, targets, preds: [B, 1, D, H, W]
    """
    idx = 0  # 取 batch 中第一个 patch
    input_patch = inputs[idx, 0].detach().cpu().numpy()   # [D, H, W]
    target_patch = targets[idx, 0].detach().cpu().numpy()
    pred_patch = preds[idx, 0].detach().cpu().numpy()

    z = input_patch.shape[0] // 2  # 取中间一层
    img = input_patch[:, :, z]
    gt = target_patch[:, :, z]
    pred = pred_patch[:, :, z]

    # 可视化
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title(f'{tag} - Input')

    axs[1].imshow(gt, cmap='gray')
    axs[1].set_title(f'{tag} - Ground Truth')

    axs[2].imshow(pred, cmap='gray')
    axs[2].set_title(f'{tag} - Prediction')

    for ax in axs:
        ax.axis('off')

    # 上传到 wandb（或你也可以选择保存本地）
    wandb.log({f"{tag}_patch_vis": wandb.Image(fig)}, step=step)
    plt.close(fig)
