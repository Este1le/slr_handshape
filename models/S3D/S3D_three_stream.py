import torch
from torch import nn
import torch.nn.functional as F

#from .fusion import Lateral_Conn
from .S3D import S3D_backbone


class S3D_three_stream(nn.Module):
    def __init__(self, use_block=4, freeze_block=(1,0,1), downsample=2, pose_inchannels=17, 
        #**kwargs
        ):
        super(S3D_three_stream, self).__init__()
        # need to set channel dimension
        self.rgb_stream = S3D_backbone(3, use_block, freeze_block[0], downsample)
        self.pose_stream = S3D_backbone(pose_inchannels, use_block, freeze_block[1], downsample)
        self.hs_stream = S3D_backbone(3, use_block, freeze_block[2], downsample)
        self.use_block = use_block

        # identify the index of each stage(block)
        # NOTE: As posec3d, no fusion in the final stage
        self.block_idx = [0, 3, 6, 12]  #block outputs index
        assert use_block == 4
        
    def forward(self, x_rgb, x_pose, x_hs, sgn_lengths=None):
        B, C, T_in, H, W = x_rgb.shape
        rgb_fea_lst, pose_fea_lst, hs_fea_lst = [], [], []
        for i, (rgb_layer, pose_layer, hs_layer) in enumerate(zip(self.rgb_stream.backbone.base, 
                                                        self.pose_stream.backbone.base,
                                                        self.hs_stream.backbone.base)):
            x_rgb = rgb_layer(x_rgb)
            x_pose = pose_layer(x_pose)
            x_hs = hs_layer(x_hs)

            if i in self.block_idx:
                rgb_fea_lst.append(x_rgb)
                pose_fea_lst.append(x_pose)
                hs_fea_lst.append(x_hs)
        diff = 1
        for i in range(len(rgb_fea_lst)):
            rgb_fea_lst[i] = rgb_fea_lst[i].mean(dim=(-2,-1)).permute(0,2,1)
            pose_fea_lst[i] = pose_fea_lst[i].mean(dim=(-2,-1)).permute(0,2,1)
            hs_fea_lst[i] = hs_fea_lst[i].mean(dim=(-2,-1)).permute(0,2,1)

        sgn_mask_lst, valid_len_out_lst = [], []
        rgb_out = pose_out = None
        rgb_out = rgb_fea_lst[-1]
        pose_out = pose_fea_lst[-1]
        hs_out = hs_fea_lst[-1]
        for fea in pose_fea_lst:
            B, T_out, _ = fea.shape
            sgn_mask = torch.zeros([B,1,T_out], dtype=torch.bool, device=fea.device)
            valid_len_out = torch.floor(sgn_lengths*T_out/T_in+1e-3).long() 
            for bi in range(B):
                sgn_mask[bi, :, :valid_len_out[bi]] = True
            sgn_mask_lst.append(sgn_mask)
            valid_len_out_lst.append(valid_len_out)

        return {'sgn_feature': rgb_out, 'pose_feature': pose_out, 'hs_feature': hs_out,
                'sgn_mask': sgn_mask_lst[diff:], 'valid_len_out': valid_len_out_lst[diff:],
                'rgb_fea_lst': rgb_fea_lst[diff:], 'pose_fea_lst': pose_fea_lst[diff:],
                'hs_fea_lst': hs_fea_lst[diff:]}
