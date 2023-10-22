import torch
from torch import nn
import torch.nn.functional as F

#from .fusion import Lateral_Conn
from .S3D import S3D_backbone


class S3D_two_stream_v2(nn.Module):
    def __init__(self, use_block=4, freeze_block=(1,0), downsample=2, pose_inchannels=17, 
        #**kwargs
        ):
        super(S3D_two_stream_v2, self).__init__()
        # need to set channel dimension
        self.rgb_stream = S3D_backbone(3, use_block, freeze_block[0], downsample)
        self.pose_stream = S3D_backbone(pose_inchannels, use_block, freeze_block[1], downsample)
        self.use_block = use_block

        # identify the index of each stage(block)
        # NOTE: As posec3d, no fusion in the final stage
        self.block_idx = [0, 3, 6, 12]  #block outputs index
        assert use_block == 4
        
    def forward(self, x_rgb, x_pose, sgn_lengths=None):
        B, C, T_in, H, W = x_rgb.shape
        rgb_fea_lst, pose_fea_lst = [], []
        for i, (rgb_layer, pose_layer) in enumerate(zip(self.rgb_stream.backbone.base, self.pose_stream.backbone.base)):
            x_rgb = rgb_layer(x_rgb)
            x_pose = pose_layer(x_pose)

            if i in self.block_idx:
                rgb_fea_lst.append(x_rgb)
                pose_fea_lst.append(x_pose)

        diff = 1
        for i in range(len(rgb_fea_lst)):
            rgb_fea_lst[i] = rgb_fea_lst[i].mean(dim=(-2,-1)).permute(0,2,1)
            pose_fea_lst[i] = pose_fea_lst[i].mean(dim=(-2,-1)).permute(0,2,1)

        sgn_mask_lst, valid_len_out_lst = [], []
        rgb_out = pose_out = None
        rgb_out = rgb_fea_lst[-1]
        pose_out = pose_fea_lst[-1]
        for fea in pose_fea_lst:
            B, T_out, _ = fea.shape
            sgn_mask = torch.zeros([B,1,T_out], dtype=torch.bool, device=fea.device)
            valid_len_out = torch.floor(sgn_lengths*T_out/T_in+0.001).long() 
            for bi in range(B):
                sgn_mask[bi, :, :valid_len_out[bi]] = True
            sgn_mask_lst.append(sgn_mask)
            valid_len_out_lst.append(valid_len_out)

        return {'sgn_feature': rgb_out, 'pose_feature': pose_out, 'sgn_mask': sgn_mask_lst[diff:], 'valid_len_out': valid_len_out_lst[diff:],
                'rgb_fea_lst': rgb_fea_lst[diff:], 'pose_fea_lst': pose_fea_lst[diff:]}
