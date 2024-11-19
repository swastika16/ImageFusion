import os
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from utils.loss import Fusionloss
import kornia
from kornia.losses import SSIMLoss
from net import Restormer_Encoder_Simplified, Restormer_Decoder_Simplified, BaseFeatureExtraction, ModifiedDetailFeatureExtraction
from utils.dataset import H5Dataset

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Ensure necessary directories exist
os.makedirs('checkpoint/DIDF_Encoder_Simplified', exist_ok=True)
os.makedirs('checkpoint/DIDF_Decoder_Simplified', exist_ok=True)
os.makedirs('checkpoint/BaseFuseLayer', exist_ok=True)
os.makedirs('checkpoint/DetailFuseLayer', exist_ok=True)

class SubsetH5Dataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

# Configure network
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
criteria_fusion = Fusionloss()
model_str = 'CDDFuse'

# Set hyper-parameters for training
num_epochs = 1
epoch_gap = 2
lr = 1e-4
weight_decay = 0
batch_size = 4
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']

# Coefficients of the loss function
coeff_mse_loss_VF = 1.
coeff_mse_loss_IF = 1.
coeff_decomp = 2.
coeff_tv = 5.

clip_grad_norm_value = 0.01
optim_step = 20
optim_gamma = 0.5

# Model
device = 'cpu'
DIDF_Encoder = nn.DataParallel(Restormer_Encoder_Simplified()).to(device)
DIDF_Decoder = nn.DataParallel(Restormer_Decoder_Simplified()).to(device)
BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
DetailFuseLayer = nn.DataParallel(ModifiedDetailFeatureExtraction(num_layers=2)).to(device)

# Optimizer, scheduler and loss function
optimizer1 = torch.optim.Adam(DIDF_Encoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(DIDF_Decoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = torch.optim.Adam(BaseFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)
optimizer4 = torch.optim.Adam(DetailFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)

scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma)

MSELoss = nn.MSELoss()
L1Loss = nn.L1Loss()
Loss_ssim = SSIMLoss(window_size=11, reduction='mean')

# Data loader
num_samples = 50
indices = list(range(num_samples))
original_dataset = H5Dataset(r"data/MSRS_train_imgsize_128_stride_200.h5")
subset_dataset = SubsetH5Dataset(original_dataset, indices)

trainloader = DataLoader(subset_dataset,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=0)

loader = {'train': trainloader}
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

# Train
step = 0
torch.backends.cudnn.benchmark = True
prev_time = time.time()

for epoch in range(num_epochs):
    for i, (data_VIS, data_IR) in enumerate(loader['train']):
        data_VIS, data_IR = data_VIS.to(device), data_IR.to(device)
        DIDF_Encoder.train()
        DIDF_Decoder.train()
        BaseFuseLayer.train()
        DetailFuseLayer.train()

        DIDF_Encoder.zero_grad()
        DIDF_Decoder.zero_grad()
        BaseFuseLayer.zero_grad()
        DetailFuseLayer.zero_grad()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()

        if epoch < epoch_gap:  # Phase I
            feature_V = DIDF_Encoder(data_VIS)
            feature_I = DIDF_Encoder(data_IR)
            data_VIS_hat = DIDF_Decoder(feature_V)
            data_IR_hat = DIDF_Decoder(feature_I)

            mse_loss_V = 5 * Loss_ssim(data_VIS, data_VIS_hat) + MSELoss(data_VIS, data_VIS_hat)
            mse_loss_I = 5 * Loss_ssim(data_IR, data_IR_hat) + MSELoss(data_IR, data_IR_hat)

            Gradient_loss = L1Loss(kornia.filters.SpatialGradient()(data_VIS),
                                   kornia.filters.SpatialGradient()(data_VIS_hat))

            loss = coeff_mse_loss_VF * mse_loss_V + coeff_mse_loss_IF * mse_loss_I + coeff_tv * Gradient_loss

        else:  # Phase II
            feature_V = DIDF_Encoder(data_VIS)
            feature_I = DIDF_Encoder(data_IR)
            feature_F = BaseFuseLayer(feature_I + feature_V)
            feature_F = DetailFuseLayer(feature_F)
            data_Fuse = DIDF_Decoder(feature_F)

            mse_loss_V = 5 * Loss_ssim(data_VIS, data_Fuse) + MSELoss(data_VIS, data_Fuse)
            mse_loss_I = 5 * Loss_ssim(data_IR, data_Fuse) + MSELoss(data_IR, data_Fuse)

            fusionloss, _, _ = criteria_fusion(data_VIS, data_IR, data_Fuse)

            loss = fusionloss

        loss.backward()
        nn.utils.clip_grad_norm_(DIDF_Encoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(BaseFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        nn.utils.clip_grad_norm_(DetailFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
        optimizer1.step()
        optimizer2.step()
        optimizer3.step()
        optimizer4.step()

        # Determine approximate time left
        batches_done = epoch * len(loader['train']) + i
        batches_left = num_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %.10s"
            % (
                epoch,
                num_epochs,
                i,
                len(loader['train']),
                loss.item(),
                time_left,
            )
        )

    # Adjust the learning rate
    scheduler1.step()
    scheduler2.step()
    if not epoch < epoch_gap:
        scheduler3.step()
        scheduler4.step()

    for optimizer in [optimizer1, optimizer2, optimizer3, optimizer4]:
        if optimizer.param_groups[0]['lr'] <= 1e-6:
            optimizer.param_groups[0]['lr'] = 1e-6

    # Save model checkpoints
    torch.save(DIDF_Encoder.state_dict(),
               'checkpoint/DIDF_Encoder_Simplified/encoder_epoch_%d_%d.pth' % (epoch, num_epochs))
    torch.save(DIDF_Decoder.state_dict(),
               'checkpoint/DIDF_Decoder_Simplified/decoder_epoch_%d_%d.pth' % (epoch, num_epochs))
    torch.save(BaseFuseLayer.state_dict(),
               'checkpoint/BaseFuseLayer/BaseFuse_epoch_%d_%d.pth' % (epoch, num_epochs))
    torch.save(DetailFuseLayer.state_dict(),
               'checkpoint/DetailFuseLayer/DetailFuse_epoch_%d_%d.pth' % (epoch, num_epochs))

print("\nTraining completed.")
