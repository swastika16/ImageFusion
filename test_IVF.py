import os
import numpy as np
import torch
import torch.nn as nn
from net import BaseFeatureExtraction, Restormer_Encoder_Simplified, Restormer_Decoder_Simplified
from utils.Evaluator import Evaluator
from utils.img_read_save import img_save, image_read_cv2

def remove_prefix(state_dict, prefix='module.'):
    """Remove `prefix` from the keys of `state_dict`."""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            k = k[len(prefix):]
        new_state_dict[k] = v
    return new_state_dict

# Device setup
device = 'cpu'

# Initialize models
Encoder = Restormer_Encoder_Simplified().to(device)
Decoder = Restormer_Decoder_Simplified().to(device)
BaseFuseLayer = BaseFeatureExtraction(dim=64, num_heads=8).to(device)

# Modify DetailFuseLayer to match the trained model
class ModifiedDetailFeatureExtraction(nn.Module):
    def __init__(self, num_layers=2):  # Use num_layers=2 as in training
        super(ModifiedDetailFeatureExtraction, self).__init__()
        self.layers = nn.ModuleList(
            [nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) for _ in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x

DetailFuseLayer = ModifiedDetailFeatureExtraction(num_layers=2).to(device)

# Path to the checkpoint (update this to your actual checkpoint path)
ckpt_path = "checkpoint"  # Adjust this path to where your model was saved during training

# Load the checkpoint and adjust for `nn.DataParallel`
checkpoint = {
    # 'DIDF_Encoder': torch.load(os.path.join(ckpt_path, "DIDF_Encoder/encoder_epoch_0_1.pth"), map_location=device),
    # 'DIDF_Decoder': torch.load(os.path.join(ckpt_path, "DIDF_Decoder/decoder_epoch_0_1.pth"), map_location=device),
    'DIDF_Encoder': torch.load(os.path.join(ckpt_path, "DIDF_Encoder_Simplified/encoder_epoch_0_1.pth"), map_location=device),
    'DIDF_Decoder': torch.load(os.path.join(ckpt_path, "DIDF_Decoder_Simplified/decoder_epoch_0_1.pth"), map_location=device),
    'BaseFuseLayer': torch.load(os.path.join(ckpt_path, "BaseFuseLayer/BaseFuse_epoch_0_1.pth"), map_location=device),
    'DetailFuseLayer': torch.load(os.path.join(ckpt_path, "DetailFuseLayer/DetailFuse_epoch_0_1.pth"), map_location=device)
}

# Remove 'module.' prefix if it exists (necessary if the model was trained with DataParallel)
encoder_state_dict = remove_prefix(checkpoint['DIDF_Encoder'])
decoder_state_dict = remove_prefix(checkpoint['DIDF_Decoder'])
base_fuse_layer_state_dict = remove_prefix(checkpoint['BaseFuseLayer'])
detail_fuse_layer_state_dict = remove_prefix(checkpoint['DetailFuseLayer'])

# Load the model state dictionaries
Encoder.load_state_dict(encoder_state_dict)
Decoder.load_state_dict(decoder_state_dict)
BaseFuseLayer.load_state_dict(base_fuse_layer_state_dict)
DetailFuseLayer.load_state_dict(detail_fuse_layer_state_dict)

# Set models to evaluation mode
Encoder.eval()
Decoder.eval()
BaseFuseLayer.eval()
DetailFuseLayer.eval()

# Define the dataset names and paths
for dataset_name in ["RoadScene", "TNO", "MRI_CT"]:
    print("\n" * 2 + "=" * 80)
    model_name = "CDDFuse"
    print("The test result of " + dataset_name + ' :')

    test_folder = os.path.join('test_img', dataset_name)
    test_out_folder = os.path.join('test_result', dataset_name)

    with torch.no_grad():
        for img_name in os.listdir(os.path.join(test_folder, "ir")):
            # Read and preprocess images
            data_IR = image_read_cv2(os.path.join(test_folder, "ir", img_name), mode='GRAY')[
                          np.newaxis, np.newaxis, ...] / 255.0
            data_VIS = image_read_cv2(os.path.join(test_folder, "vi", img_name), mode='GRAY')[
                           np.newaxis, np.newaxis, ...] / 255.0

            data_IR, data_VIS = torch.FloatTensor(data_IR), torch.FloatTensor(data_VIS)
            data_VIS, data_IR = data_VIS.to(device), data_IR.to(device)

            # Forward pass through the network
            # when using the simplified encoder/ decoder
            output = Encoder(data_VIS)
            feature_I = Encoder(data_IR)
            data_Fuse = Decoder(data_VIS)

            # Uncomment when using the unsimplified Encoder/ Decoder
            # feature_V_B, feature_V_D, feature_V = Encoder(data_VIS)
            # feature_I_B, feature_I_D, feature_I = Encoder(data_IR)
            # feature_F_B = BaseFuseLayer(feature_V_B + feature_I_B)
            # feature_F_D = DetailFuseLayer(feature_V_D + feature_I_D)
            # data_Fuse, _ = Decoder(data_VIS, feature_F_B, feature_F_D)
            data_Fuse = (data_Fuse - torch.min(data_Fuse)) / (torch.max(data_Fuse) - torch.min(data_Fuse))
            fi = np.squeeze((data_Fuse * 255).cpu().numpy())

            # Save the fused image
            img_save(fi, img_name.split(sep='.')[0], test_out_folder)

    # Evaluate results
    eval_folder = test_out_folder
    ori_img_folder = test_folder

    metric_result = np.zeros((8))
    for img_name in os.listdir(os.path.join(ori_img_folder, "ir")):
        ir = image_read_cv2(os.path.join(ori_img_folder, "ir", img_name), 'GRAY')
        vi = image_read_cv2(os.path.join(ori_img_folder, "vi", img_name), 'GRAY')
        fi = image_read_cv2(os.path.join(eval_folder, img_name.split('.')[0] + ".png"), 'GRAY')
        metric_result += np.array([Evaluator.EN(fi), Evaluator.SD(fi), Evaluator.SF(fi), Evaluator.MI(fi, ir, vi),
                                   Evaluator.SCD(fi, ir, vi), Evaluator.VIFF(fi, ir, vi), Evaluator.Qabf(fi, ir, vi),
                                   Evaluator.SSIM(fi, ir, vi)])

    metric_result /= len(os.listdir(eval_folder))
    print("\t\t EN\t SD\t SF\t MI\tSCD\tVIF\tQabf\tSSIM")
    print(model_name + '\t' + str(np.round(metric_result[0], 2)) + '\t'
          + str(np.round(metric_result[1], 2)) + '\t'
          + str(np.round(metric_result[2], 2)) + '\t'
          + str(np.round(metric_result[3], 2)) + '\t'
          + str(np.round(metric_result[4], 2)) + '\t'
          + str(np.round(metric_result[5], 2)) + '\t'
          + str(np.round(metric_result[6], 2)) + '\t'
          + str(np.round(metric_result[7], 2))
          )
    print("=" * 80)
