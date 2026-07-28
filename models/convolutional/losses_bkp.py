import torch
from torch import nn
import torchvision
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import logging


from utils.train_utils import calc_psnr

accelerator = "cuda" if torch.cuda.is_available() else "cpu"


# QUESTA METRICA QUI E' MEGLIO CHE ABBIA COME DATA RANGE IL RANGE DEI VALORI DEL TRAIN -> nel codcie che calocola e salva mean e sd di train/test/val aggiungere MIN-MAX
def masked_psnr(pred, gt, mask):
    masked_pred = pred[~mask]
    masked_y = gt[~mask]
    # psnr = PeakSignalNoiseRatio().to(accelerator)
    psnr = PeakSignalNoiseRatio(data_range=1).to(pred.device)
    #return calc_psnr(masked_pred, masked_y)
    return psnr(masked_pred, masked_y)
####################################

def masked_ssim(pred, gt, mask):
    # MODIFICA
    # pred[mask] = 0
    # gt[mask] = 0
    pred = pred.masked_fill(mask, 0)
    gt = gt.masked_fill(mask, 0)
    ###
    ssim = StructuralSimilarityIndexMeasure().to(pred.device)
    return ssim(pred, gt)

def masked_rmse(pred, gt, mask, stat=None):
    if stat is None:
        stat = [0, 1]
    masked_pred = (pred[~mask] * stat[1]) + stat[0]
    masked_y = (gt[~mask] * stat[1]) + stat[0]
    return torch.sqrt(torch.mean((masked_pred - masked_y) ** 2))

def masked_mse(pred, gt, mask):
    return torch.mean((pred[~mask] - gt[~mask]) ** 2)

# MODIFICA 
# RMSE SU EXP(PRED CALCOLATI SU DATI LOG)
def masked_rmse_exp_log(pred, gt, mask, stat=None):
    if stat is None:
        stat = [0, 1]
    masked_pred_exp = torch.exp(pred[~mask] * stat[1] + stat[0])
    masked_y_exp =torch.exp(gt[~mask] * stat[1] + stat[0])
    return torch.sqrt(torch.mean((masked_pred_exp - masked_y_exp) ** 2))



# RMSE SU LOG(PRED CALCOLATI SU DATI NO-LOG + THRESHOLD)
# Liste globali
mins = []
counts = []

def masked_rmse_log(pred, gt, mask, stat=None, threshold = None):
    if stat is None:
        stat = [0, 1]

    pred = pred.clone()
    gt = gt.clone()

    # 1. denormalizzo
    pred[~mask] = pred[~mask] * stat[1] + stat[0]
    gt[~mask] = gt[~mask] * stat[1] + stat[0]

    # 2. guardo che valori minimi ho e quanti valori negativi o zero ho nelle previsioni
    mins.append(pred[~mask].min().item())
    counts.append((pred[~mask] <= 0).sum().item())

    # applico threshold prima di applicare il log in modo da evitare nan --> SOLO SE SO CHE CI SONO VALORI NEGATIVI??
    if threshold is not None:
        gt[~mask] = torch.clamp(gt[~mask], min=threshold)
        pred[~mask] = torch.clamp(pred[~mask], min=threshold)

    # trasformo in log
    masked_pred_log = torch.log(pred[~mask])
    masked_y_log = torch.log(gt[~mask])   
    # print(torch.sqrt(torch.mean((masked_pred_log - masked_y_log) ** 2))) 

    # calcolo l'rmse 
    return torch.sqrt(torch.mean((masked_pred_log - masked_y_log) ** 2))

##################

    


#to try both as a penalty and by itself
class WeightedRMSELoss(nn.Module):
    '''
    This loss function takes as input a certain mask, e.g., the average of a certain variable over time for each pixel
    and weights more errors on values which are at a high distance from the mask. Here, data are normalized, hence the
    average will be approximately a tensor of zeros.
    '''
    def __init__(self):
        super(WeightedRMSELoss, self).__init__()

    def forward(self, predicted, target, mask):
        masked_pred = predicted[~mask]
        masked_y = target[~mask]
        squared_error = (masked_pred - masked_y).pow(2)
        # Calculate distances from the mask
        # The power of y values is the squared distance from the zeros tensor
        weighted_distances = (masked_y).pow(2)
        # Weighted mean squared error
        loss = torch.mean(squared_error * weighted_distances)
        return loss

def _3d_vgg_channel_loss(idx, input, target, blocks, resize, transform, feature_layers=[0, 1, 2, 3], style_layers=[]):
        section_losses = []
        channel = input[:, idx:idx+1, :, :, :]
        channel_target = target[:, idx:idx+1, :, :, :]
        for i in range(channel.shape[2]):
            section = channel[:, :, i:i+1, :, :].squeeze(1)
            section_target = channel_target[:, :, i:i+1, :, :].squeeze(1)
            section = section.repeat(1, 3, 1, 1)
            section_target = section_target.repeat(1, 3, 1, 1)
            if resize:
                section = transform(section, mode='bilinear', size=(224, 224), align_corners=False)
                section_target = transform(section_target, mode='bilinear', size=(224, 224), align_corners=False)
            loss = 0.0
            x = section
            y = section_target
            for i, block in enumerate(blocks):
                x = block(x)
                y = block(y)
                if i in feature_layers:
                    loss += torch.nn.functional.l1_loss(x, y)
                if i in style_layers:
                    act_x = x.reshape(x.shape[0], x.shape[1], -1)
                    act_y = y.reshape(y.shape[0], y.shape[1], -1)
                    gram_x = act_x @ act_x.permute(0, 2, 1)
                    gram_y = act_y @ act_y.permute(0, 2, 1)
                    loss += torch.nn.functional.l1_loss(gram_x, gram_y)
            section_losses.append(loss)
        losses_tensor = torch.stack(section_losses)
        return torch.mean(losses_tensor)

def _2d_vgg_channel_loss(idx, input, target, blocks, resize, transform, feature_layers=[0, 1, 2, 3], style_layers=[]):
    channel = input[:, idx:idx+1, :, :]
    channel_target = target[:, idx:idx+1, :, :]
    channel = channel.repeat(1, 3, 1, 1)
    channel_target = channel_target.repeat(1, 3, 1, 1)
    #channel = (channel-self.mean) / self.std
    #channel_target = (channel_target-self.mean) / self.std
    if resize:
        channel = transform(channel, mode='bilinear', size=(224, 224), align_corners=False)
        channel_target = transform(channel_target, mode='bilinear', size=(224, 224), align_corners=False)
    loss = 0.0
    x = channel
    y = channel_target
    for i, block in enumerate(blocks):
        x = block(x)
        y = block(y)
        if i in feature_layers:
            loss += torch.nn.functional.l1_loss(x, y)
        if i in style_layers:
            act_x = x.reshape(x.shape[0], x.shape[1], -1)
            act_y = y.reshape(y.shape[0], y.shape[1], -1)
            gram_x = act_x @ act_x.permute(0, 2, 1)
            gram_y = act_y @ act_y.permute(0, 2, 1)
            loss += torch.nn.functional.l1_loss(gram_x, gram_y)
    return loss

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, n_dimensions = 2, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        if n_dimensions == 2:
            self.register_buffer("channel_loss_fn", _2d_vgg_channel_loss)
        elif n_dimensions == 3:
            self.register_buffer("channel_loss_fn", _3d_vgg_channel_loss)
        else:
            raise ValueError("n_dimensions must be either 2 or 3")
        self.register_buffer("resize", resize)

    def forward(self, input, target, mask):
        # MODIFICA
        # input[mask]=0
        # target[mask]=0
        input = input.masked_fill(mask, 0)
        target = target.masked_fill(mask, 0)
        #####
        channel_losses = []
        for i in range(input.shape[1]):
            loss = self.channel_loss_fn(i, input, target, self.blocks, self.resize, self.transform)
            channel_losses.append(loss)
        losses_tensor = torch.stack(channel_losses)
        return torch.mean(losses_tensor)

class Masked_MSELoss(nn.Module):
    def __init__(self):
        super(Masked_MSELoss, self).__init__()

    def forward(self, pred, gt, mask):
        masked_pred = pred[~mask]
        masked_y = gt[~mask]
        mse = nn.MSELoss()(masked_pred, masked_y)
        return mse

class Masked_RMSELoss(nn.Module):
    def __init__(self):
        super(Masked_RMSELoss, self).__init__()

    def forward(self, pred, gt, mask):
        masked_pred = pred[~mask]
        masked_y = gt[~mask]
        mse = nn.MSELoss()(masked_pred, masked_y)
        rmse = torch.sqrt(mse)
        return rmse
