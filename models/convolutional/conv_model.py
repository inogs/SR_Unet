import torch
import os
from torch import nn
import pytorch_lightning as pl
from models.convolutional.losses_bkp import Masked_MSELoss, Masked_RMSELoss, VGGPerceptualLoss, masked_psnr, masked_ssim, masked_rmse, masked_mse, masked_rmse_exp_log, masked_rmse_log 
from models.convolutional.networks import UNet3D_MCD


class ConvModel(pl.LightningModule):
    '''
        Loss function can be either rmse, mse, perceptual.
        The number of channels must consider just the variables (i.e., not the river channel)
    '''
    def __init__(self, main_net, n_dimensions, riv_net=False, loss='rmse', num_channels=1, riv_in_dim=None, riv_out_dim=None, lr=1e-3, stats=None, log_transform = False, threshold = None, output_path = None):
        super(ConvModel, self).__init__()

        self.save_hyperparameters()
        input_channels = num_channels + 1 if riv_net else num_channels
        if riv_net and (riv_in_dim is None or riv_out_dim is None):
            raise ValueError("riv_in_dim and riv_out_dim are required when riv_net is enabled.")

        self.main_net = UNet3D_MCD(
            input_channels=input_channels,
            output_channels=num_channels,
            riv_in_dim=riv_in_dim if riv_net else None,
            riv_out_dim=riv_out_dim if riv_net else None,
        )

        if loss == 'mse':
            self.loss = Masked_MSELoss()
        elif loss == 'rmse':
            self.loss = Masked_RMSELoss()
        elif loss == 'perceptual':
            self.loss = VGGPerceptualLoss(n_dimensions=n_dimensions)
        else:
            raise ValueError('Invalid argument for the loss function')

        # MC Dropout added - to decide whether to keep
        self.river_net = True if riv_net else None
        self.name = f"conv_model_{main_net}_{loss}"
        self.n_dimensions = n_dimensions
        self.lr = lr
        self.stats = stats
        self.log_transform = log_transform
        self.threshold = threshold
        self.output_path = output_path

    def forward(self, x, riv=None, riv_mask=None):
        x = self.main_net(x, riv)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adadelta(self.parameters())
        return optimizer
    
    # MODIFICA PER ATTIVARE IL DROPOUT NELLA VALIDATION:

    # def enable_dropout(self):
    #     for name, m in self.named_modules(): # itera per i sottomoduli della rete es. e1, e2 etc
    #         if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)): #controlla se il modulo corrente è un oggetto di tipo nn.Dropout
    #             m.train() # usa .train() invece che .eval() solo per il dropout -> lo attiva anche se si è nella validation 
    #             # controllo che funzioni
    #             print(f"{name}: training={m.training}")

    # invece che chiamare la funzione dentro validation_step
    # def on_validation_epoch_start(self):
    #     self.enable_dropout()

    #############

    def training_step(self, train_batch, batch_idx):

        if self.river_net is not None:
            x, riv, y = train_batch
        else:
            x, y = train_batch

        mask = (x > 10e3).detach() # booleano non ha bisogno del .detach()
        # MODIFICA
        #x[mask] = 0
        x = torch.where(mask, torch.zeros_like(x), x)
        ##
        x = x.detach() # se x proviene dal dataloader non servirebbe il detach perchè c'è già requires_grad = False 


        if self.river_net is not None:
            riv_mask = mask[:, 0:1, :, :].detach() #extract mask of shape (bs, 1, h, w)
            pred = self.forward(x, riv, riv_mask)
        else:
            pred = self.forward(x)

        loss = self.loss(pred, y, mask)
        self.log('train_loss', loss, 
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True) # Quando usi DDP: Sincronizza questa metrica tra tutti i processi GPU prima di loggarla
        return loss
    

    def validation_step(self, val_batch, batch_idx):
        ### MODIFICA
        # COMMENTA se vuoi tornare al validation senza dropout
        # self.enable_dropout()   # riaccende solo i dropout 
        ############

        if self.river_net is not None:
            x, riv, y = val_batch
        else:
            x, y = val_batch

        mask = (x > 10e3)
        # MODIFICA
        # x[mask] = 0 
        x = torch.where(mask, torch.zeros_like(x), x)
        ####


        if self.river_net is not None:
            riv_mask = mask[:, 0:1, :, :] #extract mask of shape (bs, 1, h, w)
            pred = self.forward(x, riv, riv_mask)
        else:
            pred = self.forward(x)

        loss = self.loss(pred, y, mask)
        # psnr_score = masked_psnr(pred, y, mask)
        #MODIFICA
        self.log('val_loss', loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True,
                 sync_dist=True)
        # self.log('val_psnr', psnr_score,
        #          on_step=False,
        #          on_epoch=True,
        #          prog_bar =True,
        #          sync_dist=True)


    def on_test_start(self):
        from models.convolutional.losses_bkp import mins, counts
        mins.clear()
        counts.clear()

        self.test_rmse_values = []
        self.test_mse_values = [] 
        self.test_ssim_values = []
        self.test_rmse_log_values = []
        if self.log_transform:
            self.test_exp_rmse = []

    def test_step(self, test_batch, batch_idx):

        if self.river_net is not None:
            x, riv, y = test_batch
        else:
            x, y = test_batch

        mask = (x > 10e3)
        x = torch.where(mask, torch.zeros_like(x), x)

        if self.river_net is not None:
            riv_mask = mask[:, 0:1, :, :] #extract mask of shape (bs, 1, h, w)
            pred = self.forward(x, riv, riv_mask)
        else:
            pred = self.forward(x)

        # psnr_score = masked_psnr(pred, y, mask)
        ssim_score = masked_ssim(pred, y, mask)
        mse_score = masked_mse(pred, y, mask)

        self.test_mse_values.append(mse_score.detach())
        self.test_ssim_values.append(ssim_score.detach())

        if self.stats is not None:
            rmse_score = masked_rmse(pred, y, mask, self.stats)
            rmse_log_scores = masked_rmse_log(pred = pred, gt = y, mask = mask, stat = self.stats, threshold = self.threshold)
            # MODIFICA PER AGGIUNGERE CALCOLO SD
            self.test_rmse_values.append(rmse_score.detach())
            self.log('test_rmse', rmse_score, sync_dist=True) # defoult on_epoch=True -> il valore finale è la media delle rmse per ogni batch, batch che nel test è formato da un solo campione
            
            if not self.log_transform:
                self.test_rmse_log_values.append(rmse_log_scores.detach())
                self.log('test_rmse_log', rmse_log_scores, sync_dist=True)
            
            if self.log_transform:
                rmse_score_on_exp = masked_rmse_exp_log(pred, y, mask, self.stats)
                self.test_exp_rmse.append(rmse_score_on_exp.detach())
                self.log('test_rmse_on_exp_pred_log', rmse_score_on_exp, sync_dist=True)
        # self.log('test_psnr', psnr_score, sync_dist=True)
        self.log('test_ssim', ssim_score, sync_dist=True)
        self.log('test_mse', mse_score, sync_dist=True)

    def on_test_epoch_end(self):
        self.log("test_rmse_std", torch.stack(self.test_rmse_values).std(), sync_dist=True)
        
        if not self.log_transform:
            self.log("test_rmse_log_std", torch.stack(self.test_rmse_log_values).std(), sync_dist=True)
        
        if self.log_transform:
            self.log("test_exp_rmse_std", torch.stack(self.test_exp_rmse).std(), sync_dist=True)
        
        self.log("test_ssim_std", torch.stack(self.test_ssim_values).std(), sync_dist=True)
        self.log("test_mse_std", torch.stack(self.test_mse_values).std(), sync_dist=True)


        from models.convolutional.losses_bkp import mins, counts
        import json
        
        file_path = os.path.join(self.output_path, "mins_counts.json")

        self.print("OUTPUT PATH:", self.output_path)
        self.print("FILE JSON:", os.path.join(self.output_path, "mins_counts.json"))

        with open(file_path, "w") as f:
            json.dump(
                {
                    "mins": mins,
                    "counts": counts
                },
                f,
                indent=4
            )


