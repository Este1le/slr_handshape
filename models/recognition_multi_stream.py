import math, os
from copy import deepcopy
import random, torchvision
import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
from itertools import groupby

from .S3D.S3D import S3D_backbone
from .S3D.S3D_two_stream import S3D_two_stream_v2
from .S3D.S3D_three_stream import S3D_three_stream
from utils.misc import get_logger, neq_load_customized
from .Tokenizer import GlossTokenizer_S2G
from .Tokenizer import HandshapeTokenizer_S2G
from .Visualhead import VisualHead
from utils.gen_gaussian import gen_gaussian_hmap_op
from .module import SentenceEmbedding

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

ctc_ext_beam_search_decoder_ops = load_library.load_op_library(
    resource_loader.get_path_to_datafile('/exp/xzhang/slt/multi_cue/ctc-beam-search-op/tensorflow_ctc_ext_beam_search_decoder/python/ops/_ctc_ext_beam_search_decoder_ops.so'))
ctc_beam_extend_search_decoder = ctc_ext_beam_search_decoder_ops.ctc_ext_beam_search_decoder

def best_ctc_gls_hs_path(gls_best_paths, tf_hs_logits, hs_seg_ids, pad_id, blank_id, expand=True):
    hs_best_paths = []
    for si in range(len(gls_best_paths)):
        gls_best_path = gls_best_paths[si]
        tf_hs_logit = tf_hs_logits[si]
        hs_seg_id = hs_seg_ids[si]
        hs_best_path = []
        hs_id = 0
        start_id = 0
        for g in range(1, len(gls_best_path)):
            gls = gls_best_path[g]
            if gls!=gls_best_path[g-1] or g==len(gls_best_path)-1:
                # when gls is different from previous gls
                # or it's the last gls
                if g == len(gls_best_path) - 1:
                    end_id = g + 1
                else:
                    end_id = g
                if g==len(gls_best_path)-1 and gls == pad_id:
                    hs_best_path.extend([pad_id]*(end_id-start_id))
                    break
                if len(hs_seg_id) == 0:
                    hs_labels = torch.tensor([pad_id])
                else:
                    hs_labels = hs_seg_id[hs_id]
                while len(hs_labels) > end_id-start_id:
                    hs_labels = hs_labels[:-1]
                if len(hs_seg_id) == 0:
                    best_sub_path = torch.ones(1, end_id-start_id) * pad_id
                else:
                    best_sub_path = best_ctc_path(tf_hs_logit[start_id:end_id, :].unsqueeze(0), [torch.tensor(hs_labels)], [end_id-start_id], pad_id, blank_id, expand=expand)
                hs_best_path.extend(best_sub_path[0].tolist())
                start_id = g
                hs_id += 1
        hs_best_paths.append(hs_best_path)
    return torch.tensor(hs_best_paths).to(tf_hs_logits.device)

def best_ctc_path(tf_gloss_logits, ref_labels, input_lengths, pad_id, blank_id, expand=True):
    def get_trellis(emission, tokens, num_frame, blank_id):
        num_tokens = len(tokens)

        # Trellis has extra diemsions for both time axis and tokens.
        # The extra dim for tokens represents <SoS> (start-of-sentence)
        # The extra dim for time axis is for simplification of the code.
        trellis = torch.empty((num_frame + 1, num_tokens + 1)).to(emission.device)
        trellis[0, 0] = 0
        trellis[1:, 0] = torch.cumsum(emission[:, 0], 0)
        trellis[0, -num_tokens:] = -float("inf")
        trellis[-num_tokens:, 0] = float("inf")
        for t in range(num_frame):
            trellis[t + 1, 1:] = torch.maximum(
                # Score for staying at the same token
                trellis[t, 1:] + emission[t, blank_id],
                # Score for changing to the next token
                trellis[t, :-1] + emission[t, tokens],
            )
        return trellis        
    def backtrack(trellis, emission, tokens, blank_id):
        # Note:
        # j and t are indices for trellis, which has extra dimensions
        # for time and tokens at the beginning.
        # When referring to time frame index `T` in trellis,
        # the corresponding index in emission is `T-1`.
        # Similarly, when referring to token index `J` in trellis,
        # the corresponding index in transcript is `J-1`.
        j = trellis.size(1) - 1
        t_start = torch.argmax(trellis[:, j]).item()

        path = []
        for t in range(t_start, 0, -1):
            # 1. Figure out if the current position was stay or change
            # Note (again):
            # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
            # Score for token staying the same from time frame J-1 to T.
            stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
            # Score for token changing from C-1 at T-1 to J at T.
            changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

            # 2. Store the path with frame-wise probability.
            prob = emission[t - 1, tokens[j - 1] if changed > stayed else 0].exp().item()
            # Return token index and time index in non-trellis coordinate.
            path.append((j - 1, t - 1, prob)) # token_index: int, time_index: int, score: float

            # 3. Update the token
            if changed > stayed:
                j -= 1
                if j == 0:
                    break
        else:
            raise ValueError("Failed to align")
        return path[::-1]
    best_paths = []
    new_ref_labels = []
    for si in range(tf_gloss_logits.shape[0]):
        tokens = ref_labels[si]
        if len(tokens) == 0:
            tokens = torch.tensor([blank_id])
        tokens = tokens[tokens!=pad_id] # remove padding
        while len(tokens) > input_lengths[si]:
            tokens = tokens[:-1]
        emission = tf_gloss_logits[si, :input_lengths[si], :]
        trellis = get_trellis(emission, tokens, input_lengths[si], blank_id)
        best_path =  backtrack(trellis, emission, tokens, blank_id)
        alignment = torch.zeros(emission.size(0)).to(emission.device)
        for b in best_path:
            alignment[b[1]] = tokens[b[0]]
        best_paths.append(alignment.tolist())
        new_ref_labels.append(tokens)
    if expand:
        best_seq_paths = expand_ctc_labels(tf_gloss_logits, new_ref_labels, input_lengths, blank_id, best_paths)
    else:
        best_seq_paths = best_paths
    padded_best_seq_paths = []
    for bs in best_seq_paths:
        bs.extend([pad_id] * (max(input_lengths)-len(bs)))
        padded_best_seq_paths.append(bs)
    res_best_seq_paths = torch.tensor(padded_best_seq_paths).to(emission.device)
    return res_best_seq_paths

def expand_ctc_labels(tf_gloss_logits, decoded_gloss_sequences, input_lengths, blank_id=0, tmp_frame_gloss_sequences=None, ctc_decode=None):
    if tmp_frame_gloss_sequences==None and ctc_decode!=None:
        tmp_frame_gloss_sequences = [[] for i in range(input_lengths.shape[0])]
        for (value_idx, dense_idx) in enumerate(ctc_decode[3][0]):
            tmp_frame_gloss_sequences[dense_idx[0]].append(
                ctc_decode[4][0][value_idx].numpy() + 1
            )
    decoded_frame_gloss_sequences = []
    for si in range(len(tmp_frame_gloss_sequences)):
        tmp_seq = tmp_frame_gloss_sequences[si]
        input_length = input_lengths[si]
        gloss_logits = tf_gloss_logits[si,:input_length,:]
        mid_seq = []
        nonblank = decoded_gloss_sequences[si].tolist()
        if len(nonblank) == 1:
            mid_seq = [nonblank[0]] * len(tmp_seq)
            decoded_frame_gloss_sequences.append(mid_seq)
            continue
        pre = 0
        next = 1
        extend_next = False
        for i in range(input_length):
            if tmp_seq[i] == nonblank[pre] and mid_seq==[]:
               mid_seq.extend([nonblank[pre]] * (i+1))
            elif mid_seq != [] and tmp_seq[i] != blank_id:
                if tmp_seq[i] == nonblank[pre]:
                    mid_seq.append(nonblank[pre])
                else:
                    extend_next = False
                    pre += 1
                    next += 1
                    mid_seq.append  (nonblank[pre])
                    if next == len(nonblank):
                        mid_seq.extend([nonblank[pre]] *(input_lengths[si]-i-1))
                        break
            elif mid_seq != [] and tmp_seq[i] == blank_id:
                if gloss_logits[i, nonblank[pre]] >= gloss_logits[i, nonblank[next]] and extend_next==False:
                    mid_seq.append(nonblank[pre])
                else:
                    mid_seq.append(nonblank[next])
                    extend_next = True
        decoded_frame_gloss_sequences.append(mid_seq)
    return decoded_frame_gloss_sequences
            
def ctc_decode_func(tf_gloss_logits, input_lengths, beam_size):
    ctc_decode, _ = tf.nn.ctc_beam_search_decoder(
        inputs=tf_gloss_logits, 
        sequence_length=input_lengths.cpu().detach().numpy(),
        beam_width=beam_size,
        top_paths=1,
    )
    ctc_decode = ctc_decode[0]
    tmp_gloss_sequences = [[] for i in range(input_lengths.shape[0])]
    for (value_idx, dense_idx) in enumerate(ctc_decode.indices):
        tmp_gloss_sequences[dense_idx[0]].append(
            ctc_decode.values[value_idx].numpy() + 1
        )
    decoded_gloss_sequences = []
    for seq_idx in range(0, len(tmp_gloss_sequences)):
        decoded_gloss_sequences.append(
            [x[0] for x in groupby(tmp_gloss_sequences[seq_idx])]
        )
    return decoded_gloss_sequences

class RecognitionNetwork(torch.nn.Module):
    def __init__(self, cfg, input_type, transform_cfg, 
        input_streams=['rgb']) -> None:
        super().__init__()
        logger = get_logger()
        self.cfg = cfg
        self.input_type = input_type
        self.gloss_tokenizer = GlossTokenizer_S2G(
            cfg['GlossTokenizer'])
        if 'handshape' in self.cfg:
            self.handshapes = self.cfg['handshape']
        else:
            self.handshapes = []
        if 'right' in self.handshapes:
            self.handshape_tokenizer_right = HandshapeTokenizer_S2G(
                cfg['HandshapeTokenizer'], 'right'
            )
        else:
            self.handshape_tokenizer_right = None
        if 'left' in self.handshapes:
            self.handshape_tokenizer_left = HandshapeTokenizer_S2G(
                cfg['HandshapeTokenizer'], 'left'
            )
        else:
            self.handshape_tokenizer_left = None
        self.input_streams = input_streams
        self.fuse_method = cfg.get('fuse_method', 'empty')
        self.heatmap_cfg = cfg.get('heatmap_cfg',{})
        self.transform_cfg = transform_cfg
        self.preprocess_chunksize = cfg.get('preprocess_chunksize', 16)
        if self.input_type=='video':
            if 'rgb' in input_streams and not 'keypoint' in input_streams and not 'handshape' in input_streams:
                if 's3d' in cfg:
                    self.visual_backbone = S3D_backbone(in_channel=3, **cfg['s3d'])
                else:
                    raise ValueError
                self.visual_backbone_keypoint, self.visual_backbone_twostream = None, None
                self.visual_backbone_handshape, self.visual_backbone_threestream = None, None
            elif 'keypoint' in input_streams and not 'rgb' in input_streams and not 'handshape' in input_streams:
                if 'keypoint_s3d' in cfg:
                    self.visual_backbone_keypoint = S3D_backbone(\
                        **cfg['keypoint_s3d'])
                self.visual_backbone, self.visual_backbone_twostream = None, None
                self.visual_backbone_handshape, self.visual_backbone_threestream = None, None
            elif 'handshape' in input_streams and not 'rgb' in input_streams and not 'keypoint' in input_streams:
                if 'handshape_s3d' in cfg:
                    self.visual_backbone_handshape = S3D_backbone(in_channel=3, **cfg['handshape_s3d'])
                self.visual_backbone, self.visual_backbone_twostream = None, None
                self.visual_backbone_keypoint, self.visual_backbone_threestream = None, None
            elif 'rgb' in input_streams and 'keypoint' in input_streams and 'handshape' not in input_streams: 
                self.visual_backbone_twostream = S3D_two_stream_v2(
                    use_block=cfg['s3d']['use_block'],
                    freeze_block=(cfg['s3d']['freeze_block'], cfg['keypoint_s3d']['freeze_block']),
                    pose_inchannels=cfg['keypoint_s3d']['in_channel'])
                self.visual_backbone, self.visual_backbone_keypoint = None, None
                self.visual_backbone_handshape, self.visual_backbone_threestream = None, None
            elif 'rgb' in input_streams and 'keypoint' not in input_streams and 'handshape' in input_streams: 
                self.visual_backbone_twostream = S3D_two_stream_v2(
                    use_block=cfg['s3d']['use_block'],
                    freeze_block=(cfg['s3d']['freeze_block'], cfg['handshape_s3d']['freeze_block']),
                    pose_inchannels=3)
                self.visual_backbone, self.visual_backbone_keypoint = None, None
                self.visual_backbone_handshape, self.visual_backbone_threestream = None, None
            elif 'rgb' in input_streams and 'keypoint' in input_streams and 'handshape' in input_streams:
                self.visual_backbone_threestream = S3D_three_stream(
                    use_block=cfg['s3d']['use_block'],
                    freeze_block=(cfg['s3d']['freeze_block'], cfg['keypoint_s3d']['freeze_block'], cfg['handshape_s3d']['freeze_block']),
                    pose_inchannels=cfg['keypoint_s3d']['in_channel'])
                self.visual_backbone, self.visual_backbone_keypoint, self.visual_backbone_handshape = None, None, None
                self.visual_backbone_twostream = None
            else:
                raise ValueError

        if 'visual_head' in cfg:
            if 'rgb' in input_streams:
                #cfg['visual_head']['input_size'] = 832
                self.visual_head = VisualHead(cls_num=len(self.gloss_tokenizer), **cfg['visual_head'])
                if cfg.get('handshape_heads', False) == True: # if -1, attach handshape heads to each stream
                    if 'right' in self.handshapes:
                        self.visual_head_handshape_right = VisualHead(cls_num=len(self.handshape_tokenizer_right), **cfg['visual_head'])
                    if 'left' in self.handshapes:
                        self.visual_head_handshape_left = VisualHead(cls_num=len(self.handshape_tokenizer_left), **cfg['visual_head'])
            else:
                self.visual_head = None
            
            if 'keypoint' in input_streams:
                #cfg['visual_head']['input_size'] = 832
                self.visual_head_keypoint = VisualHead(cls_num=len(self.gloss_tokenizer), **cfg['visual_head'])
                if cfg.get('handshape_heads', False) == True:
                    if 'right' in self.handshapes:
                        self.visual_head_keypoint_handshape_right = VisualHead(cls_num=len(self.handshape_tokenizer_right), **cfg['visual_head'])
                    if 'left' in self.handshapes:
                        self.visual_head_keypoint_handshape_left = VisualHead(cls_num=len(self.handshape_tokenizer_left), **cfg['visual_head'])
            else:
                self.visual_head_keypoint = None

            if 'handshape_visual_head' in cfg:
                hvh = cfg['handshape_visual_head']
            else:
                hvh = cfg['visual_head']
            if 'handshape' in input_streams:
                if 'right' in self.handshapes or len(self.handshapes) < 1:
                    self.visual_head_handshape_hs_right = VisualHead(cls_num=len(self.handshape_tokenizer_right), **hvh)
                if 'left' in self.handshapes: # if -1, attach handshape heads to each stream
                    self.visual_head_handshape_hs_left = VisualHead(cls_num=len(self.handshape_tokenizer_left), **hvh)
            else:
                self.visual_head_handshape_hs_right = None
                self.visual_head_handshape_hs_left = None
            
            if 'triplehead' in self.fuse_method:
                assert ('rgb' in input_streams and 'keypoint' in input_streams) \
                    or ('rgb' in input_streams and 'handshape' in input_streams)
                new_cfg = deepcopy(cfg['visual_head'])
                if 'cat' in self.fuse_method:
                    new_cfg['input_size'] = 2*cfg['visual_head']['input_size']                
                self.visual_head_fuse = VisualHead(
                    cls_num=len(self.gloss_tokenizer), **new_cfg)
                if cfg.get('handshape_heads', False) == True:
                    if 'right' in self.handshapes:
                        self.visual_head_fuse_handshape_right = VisualHead(cls_num=len(self.handshape_tokenizer_right), **new_cfg)
                    if 'left' in self.handshapes:
                        self.visual_head_fuse_handshape_left = VisualHead(cls_num=len(self.handshape_tokenizer_left), **new_cfg)
            
            if 'fourhead' in self.fuse_method:
                assert 'rgb' in input_streams and 'keypoint' in input_streams and 'handshape' in input_streams
                new_cfg = deepcopy(cfg['visual_head'])
                if 'cat' in self.fuse_method:
                    new_cfg['input_size'] = 3*cfg['visual_head']['input_size']                
                self.visual_head_fuse = VisualHead(
                    cls_num=len(self.gloss_tokenizer), **new_cfg)
                if cfg.get('handshape_heads', False) == True:
                    if 'right' in self.handshapes:
                        self.visual_head_fuse_handshape_right = VisualHead(cls_num=len(self.handshape_tokenizer_right), **new_cfg)
                    if 'left' in self.handshapes:
                        self.visual_head_fuse_handshape_left = VisualHead(cls_num=len(self.handshape_tokenizer_left), **new_cfg)

        if 'pretrained_path_rgb' in cfg:
            load_dict = torch.load(os.path.join(cfg['pretrained_root'], cfg['pretrained_path_rgb']),map_location='cpu')['model_state']      
            backbone_dict, head_dict, fc_dict, head_remain_dict = {}, {}, {}, {}
            head_dict_right, head_dict_left = {}, {}
            for k, v in load_dict.items():
                if 'visual_backbone' in k:
                    backbone_dict[k.replace('recognition_network.visual_backbone.','')] = v
                if 'visual_head_handshape_right' in k and 'visual_head_handshape_right_remain' not in k:
                    head_dict_right[k.replace('recognition_network.visual_head_handshape_right.','')] = v
                if 'visual_head_handshape_left' in k and 'visual_head_handshape_left_remain' not in k:
                    head_dict_left[k.replace('recognition_network.visual_head_handshape_left.','')] = v
                if 'visual_head' in k and 'visual_head_handshape_right' not in k and 'visual_head_handshape_left' not in k and 'visual_head_remain' not in k:
                    head_dict[k.replace('recognition_network.visual_head.','')] = v

                if 'visual_backbone_handshape' in k:
                    backbone_dict[k.replace('recognition_network.visual_backbone_handshape.','')] = v
                if 'visual_head_handshape_hs_right' in k and 'visual_head_handshape_hs_right_remain' not in k:
                    head_dict_right[k.replace('recognition_network.visual_head_handshape_hs_right.','')] = v
                if 'visual_head_handshape_hs_left' in k and 'visual_head_handshape_hs_left_remain' not in k:
                    head_dict_left[k.replace('recognition_network.visual_head_handshape_hs_left.','')] = v
                if 'visual_head_handshape_hs_right' not in k\
                    and 'visual_head_handshape_hs_left' not in k\
                       and 'visual_head_handshape' in k and 'visual_head_handshape_remain' not in k:
                    head_dict[k.replace('recognition_network.visual_head_handshape.','')] = v

                if 'fc_layers_rgb' in k:
                    fc_dict[k.replace('recognition_network.fc_layers_rgb.','')] = v
                if 'visual_head_remain' in k:
                    head_remain_dict[k.replace('recognition_network.visual_head_remain.','')] = v
            if self.visual_backbone!=None and self.visual_backbone_twostream==None:
                neq_load_customized(self.visual_backbone, backbone_dict, verbose=True)
                neq_load_customized(self.visual_head, head_dict, verbose=True)
                if self.cfg.get('handshape_heads', False) == True:
                    for hand in self.handshapes:
                        neq_load_customized(eval(f"self.visual_head_handshape_{hand}"), eval(f"head_dict_{hand}"), verbose=True) 
                        logger.info('Load visual_head_{} for rgb from {}'.format(hand, os.path.join(cfg['pretrained_root'], cfg['pretrained_path_rgb'])))
                logger.info('Load visual_backbone and visual_head for rgb from {}'.format(os.path.join(cfg['pretrained_root'], cfg['pretrained_path_rgb'])))
            elif self.visual_backbone==None and self.visual_backbone_twostream!=None:
                neq_load_customized(self.visual_backbone_twostream.rgb_stream, backbone_dict, verbose=False)
                neq_load_customized(self.visual_head, head_dict, verbose=True)
                if self.cfg.get('handshape_heads', False) == True:
                    for hand in self.handshapes:
                        neq_load_customized(eval(f"self.visual_head_handshape_{hand}"), eval(f"head_dict_{hand}"), verbose=True)    
                        logger.info('Load visual_head_{} for rgb from {}'.format(hand, os.path.join(cfg['pretrained_root'], cfg['pretrained_path_rgb'])))   
                logger.info('Load visual_backbone_twostream.rgb_stream and visual_head for rgb from {}'.format(os.path.join(cfg['pretrained_root'], cfg['pretrained_path_rgb']))) 
            elif self.visual_backbone==None and self.visual_backbone_twostream==None and self.visual_backbone_threestream!=None:
                neq_load_customized(self.visual_backbone_threestream.rgb_stream, backbone_dict, verbose=False)
                neq_load_customized(self.visual_head, head_dict, verbose=True)
                if self.cfg.get('handshape_heads', False) == True:
                    for hand in self.handshapes:
                        neq_load_customized(eval(f"self.visual_head_handshape_{hand}"), eval(f"head_dict_{hand}"), verbose=True)
                        logger.info('Load visual_head_{} for rgb from {}'.format(hand, os.path.join(cfg['pretrained_root'], cfg['pretrained_path_rgb'])))      
                logger.info('Load visual_backbone_threestream.rgb_stream and visual_head for rgb from {}'.format(os.path.join(cfg['pretrained_root'], cfg['pretrained_path_rgb'])))
            else:
                logger.info('No rgb stream exists in the network')

        if 'keypoint' in self.input_streams and 'pretrained_path_keypoint' in cfg:
            load_dict = torch.load(os.path.join(cfg['pretrained_root'], cfg['pretrained_path_keypoint']),map_location='cpu')['model_state']
            backbone_dict, head_dict, fc_dict, head_remain_dict = {}, {}, {}, {}
            for k, v in load_dict.items():
                if 'visual_backbone_keypoint' in k:
                    backbone_dict[k.replace('recognition_network.visual_backbone_keypoint.','')] = v
                if 'visual_head_keypoint' in k and 'visual_head_keypoint_remain' not in k: #for model trained using new_code
                    head_dict[k.replace('recognition_network.visual_head_keypoint.','')] = v
                elif 'visual_head' in k and 'visual_head_keypoint_remain' not in k: #for model trained using old_code
                    head_dict[k.replace('recognition_network.visual_head.','')] = v
                elif 'visual_head_keypoint_remain' in k:
                    head_remain_dict[k.replace('recognition_network.visual_head_keypoint_remain.','')] = v
                if 'fc_layers_keypoint' in k:
                    fc_dict[k.replace('recognition_network.fc_layers_keypoint.','')] = v
            if self.visual_backbone_keypoint!=None and self.visual_backbone_twostream==None:
                neq_load_customized(self.visual_backbone_keypoint, backbone_dict, verbose=True)
                neq_load_customized(self.visual_head_keypoint, head_dict, verbose=True)
                logger.info('Load visual_backbone and visual_head for keypoint from {}'.format(os.path.join(cfg['pretrained_root'], cfg['pretrained_path_keypoint'])))
            elif self.visual_backbone_keypoint==None and self.visual_backbone_twostream!=None:
                neq_load_customized(self.visual_backbone_twostream.pose_stream, backbone_dict, verbose=False)
                neq_load_customized(self.visual_head_keypoint, head_dict, verbose=False)
                logger.info('Load visual_backbone_twostream.pose_stream and visual_head for pose from {}'.format(os.path.join(cfg['pretrained_root'], cfg['pretrained_path_keypoint'])))
            elif  self.visual_backbone_keypoint==None and self.visual_backbone_twostream == None and self.visual_backbone_threestream!=None:
                neq_load_customized(self.visual_backbone_threestream.pose_stream, backbone_dict, verbose=False)
                neq_load_customized(self.visual_head_keypoint, head_dict, verbose=False)
                logger.info('Load visual_backbone_threestream.pose_stream and visual_head for pose from {}'.format(os.path.join(cfg['pretrained_root'], cfg['pretrained_path_keypoint'])))
            else:
                logger.info('No pose stream exists in the network')
        
        if 'handshape' in self.input_streams and 'pretrained_path_handshape' in cfg:
            load_dict = torch.load(os.path.join(cfg['pretrained_root'], cfg['pretrained_path_handshape']),map_location='cpu')['model_state']      
            backbone_dict, head_dict, fc_dict, head_remain_dict = {}, {}, {}, {}
            head_dict_right, head_dict_left = {}, {}
            for k, v in load_dict.items():
                if 'visual_backbone_handshape' in k:
                    backbone_dict[k.replace('recognition_network.visual_backbone_handshape.','')] = v
                if 'visual_head_handshape_hs_right' in k and 'visual_head_handshape_hs_right_remain' not in k:
                    head_dict_right[k.replace('recognition_network.visual_head_handshape_hs_right.','')] = v
                if 'visual_head_handshape_hs_left' in k and 'visual_head_handshape_hs_left_remain' not in k:
                    head_dict_left[k.replace('recognition_network.visual_head_handshape_hs_left.','')] = v
                if 'visual_head_handshape_hs_right' not in k\
                    and 'visual_head_handshape_hs_left' not in k\
                       and 'visual_head_handshape' in k and 'visual_head_handshape_remain' not in k:
                    head_dict[k.replace('recognition_network.visual_head_handshape.','')] = v
                if 'fc_layers_handshape' in k:
                    fc_dict[k.replace('recognition_network.fc_layers_handshape.','')] = v
                if 'visual_head_handshape_remain' in k:
                    head_remain_dict[k.replace('recognition_network.visual_head_handshape_remain.','')] = v
            if self.visual_backbone_handshape!=None and self.visual_backbone_threestream==None:
                neq_load_customized(self.visual_backbone_handshape, backbone_dict, verbose=False)
                if head_dict_right == {} and head_dict_left == {}:
                    neq_load_customized(self.visual_head_handshape_hs_right, head_dict, verbose=True)
                else:
                    for hand in self.handshapes:
                        neq_load_customized(eval(f"self.visual_head_handshape_hs_{hand}"), eval(f"head_dict_{hand}"), verbose=True)
                        logger.info('Load visual_head_{} for handshape from {}'.format(hand, os.path.join(cfg['pretrained_root'], cfg['pretrained_path_handshape'])))        
                logger.info('Load visual_backbone and visual_head for handshape from {}'.format(os.path.join(cfg['pretrained_root'], cfg['pretrained_path_handshape'])))
            elif self.visual_backbone_handshape==None and self.visual_backbone_twostream!=None:
                neq_load_customized(self.visual_backbone_twostream.pose_stream, backbone_dict, verbose=False)
                if head_dict_right == {} and head_dict_left == {}:
                    neq_load_customized(self.visual_head_handshape_hs_right, head_dict, verbose=True)
                else:
                    for hand in self.handshapes:
                        neq_load_customized(eval(f"self.visual_head_handshape_hs_{hand}"), eval(f"head_dict_{hand}"), verbose=True)
                        logger.info('Load visual_head_{} for handshape from {}'.format(hand, os.path.join(cfg['pretrained_root'], cfg['pretrained_path_handshape'])))
                logger.info('Load visual_backbone_twostream.handshape_stream and visual_head for handshape from {}'.format(os.path.join(cfg['pretrained_root'], cfg['pretrained_path_handshape']))) 
            elif self.visual_backbone_handshape==None and self.visual_backbone_threestream!=None:
                neq_load_customized(self.visual_backbone_threestream.hs_stream, backbone_dict, verbose=False)
                if head_dict_right == {} and head_dict_left == {}:
                    neq_load_customized(self.visual_head_handshape_hs_right, head_dict, verbose=True)
                else:
                    for hand in self.handshapes:
                        neq_load_customized(eval(f"self.visual_head_handshape_hs_{hand}"), eval(f"head_dict_{hand}"), verbose=True)
                        logger.info('Load visual_head_{} for handshape from {}'.format(hand, os.path.join(cfg['pretrained_root'], cfg['pretrained_path_handshape'])))
                logger.info('Load visual_backbone_threestream.hs_stream and visual_head for handshape from {}'.format(os.path.join(cfg['pretrained_root'], cfg['pretrained_path_handshape'])))
            else:
                logger.info('No handshape stream exists in the network')
        
        self.reduction_method = 'sum'
        self.recognition_loss_func = torch.nn.CTCLoss(
            blank=self.gloss_tokenizer.silence_id, zero_infinity=True,
            reduction=self.reduction_method
        )

    def compute_recognition_loss(self, gloss_labels, gloss_lengths, gloss_probabilities_log, input_lengths):
        loss = self.recognition_loss_func(
            log_probs = gloss_probabilities_log.permute(1,0,2), #T,N,C
            targets = gloss_labels,
            input_lengths = input_lengths,
            target_lengths = gloss_lengths
        )
        loss = loss/gloss_probabilities_log.shape[0]
        return loss

    def decode(self, gloss_logits, beam_size, input_lengths):
        gloss_logits = gloss_logits.permute(1, 0, 2) #T,B,V
        gloss_logits = gloss_logits.cpu().detach().numpy()
        tf_gloss_logits = np.concatenate(
            (gloss_logits[:, :, 1:], gloss_logits[:, :, 0, None]),
            axis=-1,
        )
        decoded_gloss_sequences = ctc_decode_func(
            tf_gloss_logits=tf_gloss_logits,
            input_lengths=input_lengths,
            beam_size=beam_size
        )
        return decoded_gloss_sequences
    
    def get_best_alignment(self, tf_gloss_logits, ref_labels, input_lengths, pad_id, blank_id, expand=True):
        return best_ctc_path(tf_gloss_logits, ref_labels, input_lengths, pad_id, blank_id, expand=expand)

    def generate_batch_heatmap(self, keypoints):
        B,T,N,D = keypoints.shape
        keypoints = keypoints.reshape(-1, N, D)
        n_chunk = int(math.ceil((B*T)/self.preprocess_chunksize))
        chunks = torch.split(keypoints, n_chunk, dim=0)
        heatmaps = []
        for chunk in chunks:
            hm = gen_gaussian_hmap_op(
                coords=chunk,  
                **self.heatmap_cfg) 
            _, N, H, W = hm.shape
            heatmaps.append(hm)
        heatmaps = torch.cat(heatmaps, dim=0) 
        return heatmaps.reshape(B,T,N,H,W) 

    def apply_spatial_ops(self, x, spatial_ops_func):
        B, T, C_, H, W = x.shape
        x = x.view(-1, C_, H, W)
        chunks = torch.split(x, self.preprocess_chunksize, dim=0)
        transformed_x = []
        for chunk in chunks:
            transformed_x.append(spatial_ops_func(chunk))
        _, C_, H_o, W_o = transformed_x[-1].shape
        transformed_x = torch.cat(transformed_x, dim=0)
        transformed_x = transformed_x.view(B, T, C_, H_o, W_o)
        return transformed_x    

    def augment_preprocess_inputs(self, is_train, sgn_videos=None, sgn_heatmaps=None):
        rgb_h, rgb_w = self.transform_cfg.get('img_size',224), self.transform_cfg.get('img_size',224)
        if sgn_heatmaps!=None:
            hm_h, hm_w = self.heatmap_cfg['input_size'], self.heatmap_cfg['input_size']
            if sgn_videos!=None:
                rgb_h0, rgb_w0 = sgn_videos.shape[-2],sgn_videos.shape[-1] 
                hm_h0, hm_w0 = sgn_heatmaps.shape[-2],sgn_heatmaps.shape[-1]  
                factor_h, factor_w= hm_h0/rgb_h0, hm_w0/rgb_w0 
        if is_train:
            if sgn_videos!=None:
                if  self.transform_cfg.get('color_jitter',False) and random.random()<0.3:
                    color_jitter_op = torchvision.transforms.ColorJitter(0.4,0.4,0.4,0.1)
                    sgn_videos = color_jitter_op(sgn_videos)
                i,j,h,w = torchvision.transforms.RandomResizedCrop.get_params(
                    img=sgn_videos,
                    scale=(self.transform_cfg.get('bottom_area',0.2), 1.0), 
                    ratio=(self.transform_cfg.get('aspect_ratio_min',3./4), 
                        self.transform_cfg.get('aspect_ratio_max',4./3)))
                sgn_videos = self.apply_spatial_ops(
                    sgn_videos, 
                    spatial_ops_func=lambda x:torchvision.transforms.functional.resized_crop(
                        x, i, j, h, w, [rgb_h, rgb_w]))
            if sgn_heatmaps!=None:
                if sgn_videos!=None:
                    i2, j2, h2, w2 = int(i*factor_h), int(j*factor_w), int(h*factor_h), int(w*factor_w)
                else:
                    i2, j2, h2, w2 = torchvision.transforms.RandomResizedCrop.get_params(
                        img=sgn_heatmaps,
                        scale=(self.transform_cfg.get('bottom_area',0.2), 1.0), 
                        ratio=(self.transform_cfg.get('aspect_ratio_min',3./4), 
                            self.transform_cfg.get('aspect_ratio_max',4./3)))
                sgn_heatmaps = self.apply_spatial_ops(
                        sgn_heatmaps,
                        spatial_ops_func=lambda x:torchvision.transforms.functional.resized_crop(
                         x, i2, j2, h2, w2, [hm_h, hm_w]))
        else:
            if sgn_videos!=None:
                spatial_ops = []
                if self.transform_cfg.get('center_crop',False)==True:
                    spatial_ops.append(torchvision.transforms.CenterCrop(
                        self.transform_cfg['center_crop_size']))
                spatial_ops.append(torchvision.transforms.Resize([rgb_h, rgb_w]))
                spatial_ops = torchvision.transforms.Compose(spatial_ops)
                sgn_videos = self.apply_spatial_ops(sgn_videos, spatial_ops)
            if sgn_heatmaps!=None:
                spatial_ops = []
                if self.transform_cfg.get('center_crop',False)==True:
                    spatial_ops.append(
                        torchvision.transforms.CenterCrop(
                            [int(self.transform_cfg['center_crop_size']*factor_h),
                            int(self.transform_cfg['center_crop_size']*factor_w)]))
                spatial_ops.append(torchvision.transforms.Resize([hm_h, hm_w]))
                spatial_ops = torchvision.transforms.Compose(spatial_ops)
                sgn_heatmaps = self.apply_spatial_ops(sgn_heatmaps, spatial_ops)                

        if sgn_videos!=None:
            sgn_videos = sgn_videos[:,:,[2,1,0],:,:] 
            sgn_videos = (sgn_videos-0.5)/0.5
            sgn_videos = sgn_videos.permute(0,2,1,3,4).float() 
        if sgn_heatmaps!=None:
            sgn_heatmaps = (sgn_heatmaps-0.5)/0.5
            sgn_heatmaps = sgn_heatmaps.permute(0,2,1,3,4).float()
        return sgn_videos, sgn_heatmaps

    def forward(self, is_train, gloss_labels, gls_lengths,
        handshape_labels_right=None, hs_lengths_right=None,
        step=0,
        handshape_labels_left=None, hs_lengths_left=None,
        sgn_features=None, sgn_mask=None,
        sgn_videos=None, sgn_lengths=None,
        sgn_keypoints=None,
        hs_videos=None,
        hs_seg_ids_right=None,
        hs_seg_ids_left=None,
        head_rgb_input=None, head_keypoint_input=None):
        compute_ctc_loss = True
        if 'cross_entropy' in self.cfg:
            ce_strategy = self.cfg['cross_entropy'].get('strategy', 'joint')
            if ce_strategy == 'iterative' and step%2==1:
                compute_ctc_loss = False
        if is_train!=True:
            compute_ctc_loss = True
        if self.cfg.get('inference', False):
            compute_ctc_loss = False
        
        if self.input_type=='video':
            vb_outputs = []
            with torch.no_grad():
                if 'keypoint' in self.input_streams:
                    assert sgn_keypoints!=None
                    sgn_heatmaps = self.generate_batch_heatmap(
                            sgn_keypoints) 
                else:
                    sgn_heatmaps = None
                
                if not 'rgb' in self.input_streams and 'keypoint' in self.input_streams:
                    sgn_videos = None

                sgn_videos,sgn_heatmaps = self.augment_preprocess_inputs(is_train=is_train, sgn_videos=sgn_videos, sgn_heatmaps=sgn_heatmaps)
                hs_videos,_ = self.augment_preprocess_inputs(is_train=is_train, sgn_videos=hs_videos, sgn_heatmaps=None)
            if 'rgb' in self.input_streams and not 'keypoint' in self.input_streams and not 'handshape' in self.input_streams:              
                vb_outputs = self.visual_backbone(sgn_videos=sgn_videos, sgn_lengths=sgn_lengths)
            elif 'keypoint' in self.input_streams and not 'rgb' in self.input_streams:
                vb_outputs = self.visual_backbone_keypoint(sgn_videos=sgn_heatmaps, sgn_lengths=sgn_lengths)
            elif 'handshape' in self.input_streams and not 'rgb' in self.input_streams and not 'keypoint' in self.input_streams:
                if hs_videos!=None:
                    sgn_videos = hs_videos
                vb_outputs = self.visual_backbone_handshape(sgn_videos=sgn_videos, sgn_lengths=sgn_lengths)
            elif 'rgb' in self.input_streams and 'keypoint' in self.input_streams and not 'handshape' in self.input_streams:
                vb_outputs = self.visual_backbone_twostream(x_rgb=sgn_videos, x_pose=sgn_heatmaps, sgn_lengths=sgn_lengths)
            elif 'rgb' in self.input_streams and 'handshape' in self.input_streams and not 'keypoint' in self.input_streams:
                vb_outputs = self.visual_backbone_twostream(x_rgb=sgn_videos, x_pose=sgn_videos, sgn_lengths=sgn_lengths)
            elif 'rgb' in self.input_streams and 'keypoint' in self.input_streams and 'handshape' in self.input_streams:
                if hs_videos==None:
                    hs_videos = sgn_videos
                vb_outputs = self.visual_backbone_threestream(x_rgb=sgn_videos, x_pose=sgn_heatmaps, x_hs=hs_videos, sgn_lengths=sgn_lengths)
            if self.fuse_method=='empty':
                assert len(self.input_streams)==1, self.input_streams
                if 'rgb' in self.input_streams:
                    head_outputs = self.visual_head(
                        x=vb_outputs['sgn'],
                        mask=vb_outputs['sgn_mask'].squeeze(1), 
                        valid_len_in=vb_outputs['valid_len_out'][-1])
                    head_outputs['head_rgb_input'] = vb_outputs['sgn']
                    if self.cfg.get('handshape_heads', False) == True:
                        if 'right' in self.handshapes:
                            head_outputs_handshape_right = self.visual_head_handshape_right(
                                x=vb_outputs['sgn'],
                                mask=vb_outputs['sgn_mask'].squeeze(1), 
                                valid_len_in=vb_outputs['valid_len_out'][-1])
                        if 'left' in self.handshapes:
                            head_outputs_handshape_left = self.visual_head_handshape_left(
                                x=vb_outputs['sgn'],
                                mask=vb_outputs['sgn_mask'].squeeze(1), 
                                valid_len_in=vb_outputs['valid_len_out'][-1])
                        head_outputs_handshape = {}
                        for hand in self.handshapes:
                            for item in ['logits', 'probabilities_log', 'probabilities']:
                                head_outputs_handshape['rgb_handshape_'+hand+'_'+item] \
                                    = eval("head_outputs_handshape_"+hand)['gloss_'+item]
                        head_outputs.update(head_outputs_handshape)
                elif 'keypoint' in self.input_streams:
                    head_outputs = self.visual_head_keypoint(
                        x=vb_outputs['sgn'],
                        mask=vb_outputs['sgn_mask'].squeeze(1), 
                        valid_len_in=vb_outputs['valid_len_out'][-1])
                    head_outputs['head_keypoint_input'] = vb_outputs['sgn']
                elif 'handshape' in self.input_streams:
                    head_outputs = {}
                    for hand in self.handshapes:
                        head_outputs.update(eval(f"self.visual_head_handshape_hs_{hand}")(
                            x=vb_outputs['sgn'],
                            mask=vb_outputs['sgn_mask'].squeeze(1), 
                            valid_len_in=vb_outputs['valid_len_out'][-1]))
                        head_outputs['head_handhshape_input'] = vb_outputs['sgn']
                        for k in list(head_outputs.keys()):
                            if "gloss" in k:
                                head_outputs[k.replace('gloss', f'handshape_{hand}')] = head_outputs[k]
                                del head_outputs[k]
                else:
                    raise ValueError
                head_outputs['valid_len_out_lst'] = vb_outputs['valid_len_out']
            elif 'doublehead' in self.fuse_method or 'triplehead' in self.fuse_method or 'fourhead' in self.fuse_method:
                assert ('rgb' in self.input_streams and 'keypoint' in self.input_streams) or \
                    ('rgb' in self.input_streams and 'handshape' in self.input_streams)
                # rgb
                head_outputs_rgb = self.visual_head(
                    x=vb_outputs['sgn_feature'], 
                    mask=vb_outputs['sgn_mask'][-1], 
                    valid_len_in= vb_outputs['valid_len_out'][-1])
                    
                head_rgb_input = vb_outputs['sgn_feature']

                head_outputs = {'gloss_logits': None, 
                                'rgb_gloss_logits': head_outputs_rgb['gloss_logits'],
                                'gloss_probabilities_log':None,
                                'rgb_gloss_probabilities_log': head_outputs_rgb['gloss_probabilities_log'],
                                'gloss_probabilities': None,
                                'rgb_gloss_probabilities': head_outputs_rgb['gloss_probabilities'],
                                'valid_len_out': head_outputs_rgb['valid_len_out'],
                                'valid_len_out_lst': vb_outputs['valid_len_out'],
                                'head_rgb_input': head_rgb_input, 
                                }
                if self.cfg.get('handshape_heads', False) == True:
                    for hand in self.handshapes:
                        head_outputs_rgb_handshape = eval(f"self.visual_head_handshape_{hand}")(
                            x=vb_outputs['sgn_feature'],
                            mask=vb_outputs['sgn_mask'][-1], 
                            valid_len_in=vb_outputs['valid_len_out'][-1])
                        for k in list(head_outputs_rgb_handshape.keys()):
                            if "gloss" in k:
                                head_outputs_rgb_handshape[k.replace('gloss', f'rgb_handshape_{hand}')] = head_outputs_rgb_handshape[k]
                                del head_outputs_rgb_handshape[k]
                        head_outputs.update(head_outputs_rgb_handshape)
                
                # keypoint
                if "keypoint" in self.input_streams:
                    head_keypoint_input = vb_outputs['pose_feature']
                    head_outputs_keypoint = self.visual_head_keypoint(
                        x=vb_outputs['pose_feature'], 
                        mask=vb_outputs['sgn_mask'][-1], 
                        valid_len_in=vb_outputs['valid_len_out'][-1])
                    if self.cfg.get('handshape_heads', False) == True:
                        if 'right' in self.handshapes:
                            head_outputs_keypoint_handshape_right = self.visual_head_keypoint_handshape_right(
                                x=vb_outputs['pose_feature'],
                                mask=vb_outputs['sgn_mask'][-1], 
                                valid_len_in=vb_outputs['valid_len_out'][-1])
                        if 'left' in self.handshapes:
                            head_outputs_keypoint_handshape_left = self.visual_head_keypoint_handshape_left(
                                x=vb_outputs['pose_feature'],
                                mask=vb_outputs['sgn_mask'][-1], 
                                valid_len_in=vb_outputs['valid_len_out'][-1])
                    head_outputs.update({
                        'keypoint_gloss_logits': head_outputs_keypoint['gloss_logits'],
                        'keypoint_gloss_probabilities_log': head_outputs_keypoint['gloss_probabilities_log'],
                        'keypoint_gloss_probabilities': head_outputs_keypoint['gloss_probabilities'],
                        'head_keypoint_input': head_keypoint_input,
                    })
                    if self.cfg.get('handshape_heads', False) == True:
                        head_outputs_handshape = {}
                        for hand in self.handshapes:
                            for item in ['logits', 'probabilities_log', 'probabilities']:
                                head_outputs_handshape['rgb_handshape_'+hand+'_'+item] \
                                    = eval("head_outputs_handshape_"+hand)['gloss_'+item]
                                head_outputs_handshape['keypoint_handshape_'+hand+'_'+item] \
                                    = eval("head_outputs_keypoint_handshape_"+hand)['gloss_'+item]
                        head_outputs.update(head_outputs_handshape)
                if "handshape" in self.input_streams:
                    if 'triplehead' in self.fuse_method:
                        head_handshape_input = vb_outputs['pose_feature']
                    elif 'fourhead' in self.fuse_method:
                        head_handshape_input = vb_outputs['hs_feature']
                    for hand in self.handshapes:
                        head_outputs_handshape = eval(f"self.visual_head_handshape_hs_{hand}")(
                            x=head_handshape_input,
                            mask=vb_outputs['sgn_mask'][-1], 
                            valid_len_in=vb_outputs['valid_len_out'][-1])
                        for k in list(head_outputs_handshape.keys()):
                            if "gloss" in k:
                                head_outputs_handshape[k.replace('gloss', f'handshape_{hand}')] = head_outputs_handshape[k]
                                del head_outputs_handshape[k]
                        head_outputs.update(head_outputs_handshape)
                
                if 'triplehead' in self.fuse_method:
                    if 'handshape' in self.input_streams:
                        head_keypoint_input = head_handshape_input
                    assert self.visual_head_fuse!=None
                    if 'plus' in self.fuse_method:
                        fused_sgn_features = head_rgb_input+head_keypoint_input
                    elif 'cat' in self.fuse_method:
                        if self.cfg.get('cat_order', 'pose_first')=='rgb_first':
                            fused_sgn_features = torch.cat([head_rgb_input, head_keypoint_input], dim=-1)
                        else:
                            fused_sgn_features = torch.cat([head_keypoint_input, head_rgb_input], dim=-1) #B,T,D
                    else:
                        raise ValueError
                    head_outputs_fuse = self.visual_head_fuse(
                        x=fused_sgn_features, 
                        mask=vb_outputs['sgn_mask'][-1], 
                        valid_len_in=vb_outputs['valid_len_out'][-1])
                    if self.cfg.get('handshape_heads', False) == True:
                        if 'right' in self.handshapes:
                            head_outputs_handshape_right_fuse = self.visual_head_fuse_handshape_right(
                                x=fused_sgn_features, 
                                mask=vb_outputs['sgn_mask'][-1], 
                                valid_len_in=vb_outputs['valid_len_out'][-1])
                        if 'left' in self.handshapes:
                            head_outputs_handshape_left_fuse = self.visual_head_fuse_handshape_left(
                                x=fused_sgn_features, 
                                mask=vb_outputs['sgn_mask'][-1], 
                                valid_len_in=vb_outputs['valid_len_out'][-1])

                    head_outputs['fuse_gloss_probabilities'] = head_outputs_fuse['gloss_probabilities']
                    head_outputs['fuse_gloss_probabilities_log'] = head_outputs_fuse['gloss_probabilities_log']
                    head_outputs['fuse_gloss_logits'] = head_outputs_fuse['gloss_logits']
                    head_outputs['fuse_gloss_feature'] = head_outputs_fuse['gloss_feature']
                    head_outputs['head_fuse_input'] = fused_sgn_features
                    if self.cfg.get('handshape_heads', False) == True:
                        head_outputs_handshape = {}
                        for hand in self.handshapes:
                            for item in ['logits', 'probabilities_log', 'probabilities']:
                                head_outputs_handshape['fuse_handshape_'+hand+'_'+item] \
                                    = eval("head_outputs_handshape_"+hand+"_fuse")['gloss_'+item]
                        head_outputs.update(head_outputs_handshape)
                
                if 'fourhead' in self.fuse_method:
                    assert 'handshape' in self.input_streams
                    assert self.visual_head_fuse!=None
                    if 'plus' in self.fuse_method:
                        fused_sgn_features = head_rgb_input+head_keypoint_input+head_handshape_input
                    elif 'cat' in self.fuse_method:
                        if self.cfg.get('cat_order', 'pose_first')=='rgb_first':
                            fused_sgn_features = torch.cat([head_rgb_input, head_keypoint_input, head_handshape_input], dim=-1)
                        else:
                            fused_sgn_features = torch.cat([head_keypoint_input, head_rgb_input, head_handshape_input], dim=-1) #B,T,D
                    else:
                        raise ValueError
                    head_outputs_fuse = self.visual_head_fuse(
                        x=fused_sgn_features, 
                        mask=vb_outputs['sgn_mask'][-1], 
                        valid_len_in=vb_outputs['valid_len_out'][-1])
                    head_outputs['fuse_gloss_probabilities'] = head_outputs_fuse['gloss_probabilities']
                    head_outputs['fuse_gloss_probabilities_log'] = head_outputs_fuse['gloss_probabilities_log']
                    head_outputs['fuse_gloss_logits'] = head_outputs_fuse['gloss_logits']
                    head_outputs['fuse_gloss_feature'] = head_outputs_fuse['gloss_feature']
                    head_outputs['head_fuse_input'] = fused_sgn_features
                    
 
                if 'doublehead' in self.fuse_method:   
                    sum_probs = head_outputs['rgb_gloss_probabilities']+head_outputs['keypoint_gloss_probabilities']
                    head_outputs['ensemble_last_gloss_logits'] = sum_probs.log()

                elif 'triplehead' in self.fuse_method:
                    if 'keypoint' in self.input_streams:
                        head_outputs['ensemble_last_gloss_logits'] = (head_outputs['fuse_gloss_probabilities']+\
                            head_outputs['rgb_gloss_probabilities']+head_outputs['keypoint_gloss_probabilities']).log()
                    else:
                        head_outputs['ensemble_last_gloss_logits'] = (head_outputs['fuse_gloss_probabilities']+ \
                            head_outputs['rgb_gloss_probabilities']).log()
                    if 'keypoint' in self.input_streams and self.handshapes != []:
                        for hand in self.handshapes:
                            head_outputs['ensemble_last_handshape_'+hand+'_logits'] = (head_outputs['fuse_handshape_'+hand+'_probabilities']+\
                                head_outputs['rgb_handshape_'+hand+'_probabilities']+\
                                    head_outputs['keypoint_handshape_'+hand+'_probabilities']).log()
                    if 'handshape' in self.input_streams and self.cfg.get('handshape_heads', False) == True:
                        for hand in self.handshapes:
                            head_outputs['ensemble_last_handshape_'+hand+'_logits'] = (head_outputs['fuse_handshape_'+hand+'_probabilities']+\
                                head_outputs['rgb_handshape_'+hand+'_probabilities']+\
                                        head_outputs['handshape_'+hand+'_probabilities']).log()
                elif 'fourhead' in self.fuse_method:
                    head_outputs['ensemble_last_gloss_logits'] = (head_outputs['fuse_gloss_probabilities']+\
                        head_outputs['rgb_gloss_probabilities']+head_outputs['keypoint_gloss_probabilities']).log()
                    if self.cfg.get('handshape_heads', False) == True:
                        for hand in self.handshapes:
                            head_outputs['ensemble_last_handshape_'+hand+'_logits'] = (head_outputs['fuse_handshape_'+hand+'_probabilities']+\
                                head_outputs['rgb_handshape_'+hand+'_probabilities']+\
                                    head_outputs['keypoint_handshape_'+hand+'_probabilities'] +\
                                        head_outputs['handshape_'+hand+'_probabilities']).log()
                else:
                    raise ValueError 
                head_outputs['ensemble_last_gloss_probabilities_log'] = head_outputs['ensemble_last_gloss_logits'].log_softmax(2) 
                head_outputs['ensemble_last_gloss_probabilities'] = head_outputs['ensemble_last_gloss_logits'].softmax(2)
                if self.cfg.get('handshape_heads', False) == True:
                    for hand in self.handshapes:
                        head_outputs['ensemble_last_handshape_'+hand+'_probabilities_log'] = \
                            head_outputs['ensemble_last_handshape_'+hand+'_logits'].log_softmax(2) 
                        head_outputs['ensemble_last_handshape_'+hand+'_probabilities'] = \
                            head_outputs['ensemble_last_handshape_'+hand+'_logits'].softmax(2)
            else:
                raise ValueError
            valid_len_out = head_outputs['valid_len_out']          
        else:
            raise ValueError

        outputs = {**head_outputs,
            'input_lengths': valid_len_out}    
        if self.fuse_method=='empty':
            probabilities_log = head_outputs['gloss_probabilities_log'] \
                if 'gloss_probabilities_log' in head_outputs else None
            for prefix in ['rgb_', '']:
                if 'right' in self.handshapes and f'{prefix}handshape_right_probabilities_log' in head_outputs:
                    probabilities_log_right = head_outputs[f'{prefix}handshape_right_probabilities_log']
                if 'left' in self.handshapes and f'{prefix}handshape_left_probabilities_log' in head_outputs:
                    probabilities_log_left = head_outputs[f'{prefix}handshape_left_probabilities_log']
            if compute_ctc_loss:
                if 'handshape' not in self.input_streams:
                    outputs['recognition_loss'] = self.compute_recognition_loss(
                        gloss_labels=gloss_labels, gloss_lengths=gls_lengths,
                        gloss_probabilities_log=probabilities_log,
                        input_lengths=valid_len_out
                    )
                    if self.cfg.get('handshape_heads', False) == True:
                        for hand in self.handshapes:
                            outputs[f'recognition_loss_rgb_handshape_{hand}'] = self.compute_recognition_loss(
                            gloss_labels=eval(f'handshape_labels_{hand}'), gloss_lengths=eval(f'hs_lengths_{hand}'),
                            gloss_probabilities_log=head_outputs[f'rgb_handshape_{hand}_probabilities_log'],
                            input_lengths=valid_len_out)
                            outputs['recognition_loss'] += outputs[f'recognition_loss_rgb_handshape_{hand}']
                                                            #* self.cfg['handshape'][hand]
                else:
                    outputs['recognition_loss'] = 0
                    for hand in self.handshapes:
                        outputs[f'recognition_loss_{hand}'] = self.compute_recognition_loss(
                            gloss_labels=eval(f"handshape_labels_{hand}"), gloss_lengths=eval(f"hs_lengths_{hand}"),
                            gloss_probabilities_log=eval(f"probabilities_log_{hand}"),
                            input_lengths=valid_len_out
                        )
                        outputs['recognition_loss'] += outputs[f'recognition_loss_{hand}']
            self.cfg['gloss_feature_ensemble'] = self.cfg.get('gloss_feature_ensemble','gloss_feature')
            gloss_feature_ensemble = self.cfg['gloss_feature_ensemble'] if 'gloss_feature' in outputs \
                else self.cfg['gloss_feature_ensemble'].replace('gloss', 'handshape_right')
            outputs['gloss_feature'] = outputs[gloss_feature_ensemble]
        elif 'triplehead' in self.fuse_method:
            assert ('rgb' in self.input_streams and 'keypoint' in self.input_streams) \
                or ('rgb' in self.input_streams and 'handshape' in self.input_streams)
            if compute_ctc_loss:
                for k in ['rgb', 'keypoint', 'fuse']:
                    if f'{k}_gloss_probabilities_log' in head_outputs:
                        glbs=gloss_labels
                        gles=gls_lengths
                        gpl=head_outputs[f'{k}_gloss_probabilities_log']
                        ilv = valid_len_out
                        outputs[f'recognition_loss_{k}'] = self.compute_recognition_loss(
                        gloss_labels=glbs, gloss_lengths=gles,
                        gloss_probabilities_log=gpl,
                        input_lengths=ilv)
                        if self.cfg.get('handshape_heads', False) == True:
                            for hand in self.handshapes:
                                glbs=eval(f'handshape_labels_{hand}')
                                gles=eval(f'hs_lengths_{hand}')
                                gpl=head_outputs[f'{k}_handshape_{hand}_probabilities_log']
                                ilv = valid_len_out
                                outputs[f'recognition_loss_{k}_handshape_{hand}'] = self.compute_recognition_loss(
                                gloss_labels=glbs, gloss_lengths=gles,
                                gloss_probabilities_log=gpl,
                                input_lengths=ilv)
                for hand in self.handshapes:
                    hlh = eval(f'handshape_labels_{hand}') 
                    hleh = eval(f'hs_lengths_{hand}')
                    hohpl = head_outputs[f'handshape_{hand}_probabilities_log']
                    il = valid_len_out
                    if f'handshape_{hand}_probabilities_log' in head_outputs:      
                        outputs[f'recognition_loss_handshape_{hand}'] = self.compute_recognition_loss(
                            gloss_labels=hlh, gloss_lengths=hleh,
                            gloss_probabilities_log=hohpl, input_lengths=il)

            self.cfg['gloss_feature_ensemble'] = self.cfg.get('gloss_feature_ensemble','fuse_gloss_feature')
            if '@' in self.cfg['gloss_feature_ensemble']:
                feat_name, agg = self.cfg['gloss_feature_ensemble'].split('@')
                gloss_feature = [head_outputs[f'{k}_{feat_name}'] for k in ['fuse','rgb','keypoint']]
                if agg == 'cat':
                    gloss_feature = torch.cat(gloss_feature, dim=-1)
                elif agg == 'plus':
                    gloss_feature = sum(gloss_feature)
                else:
                    raise ValueError
                outputs['gloss_feature'] = gloss_feature
            else:
                stream, feat_name = self.cfg['gloss_feature_ensemble'].split('_gloss_')
                feat_name = 'gloss_'+feat_name
                outputs['gloss_feature'] = outputs[f'{stream}_{feat_name}']
            if compute_ctc_loss:
                outputs['recognition_loss'] = outputs['recognition_loss_rgb'] + outputs['recognition_loss_fuse']
                if 'keypoint' in self.input_streams:
                    outputs['recognition_loss'] += outputs['recognition_loss_keypoint']
                elif 'handshape' in self.input_streams:
                    for hand in self.handshapes:
                        outputs['recognition_loss'] += outputs[f'recognition_loss_handshape_{hand}']
                if self.cfg.get('handshape_heads', False) == True:
                    for hand in self.handshapes:
                        outputs['recognition_loss'] += outputs[f'recognition_loss_rgb_handshape_{hand}'] + \
                        outputs[f'recognition_loss_fuse_handshape_{hand}']
        
        elif 'fourhead' in self.fuse_method:
            assert 'rgb' in self.input_streams and 'keypoint' in self.input_streams and 'handshape' in self.input_streams
            if compute_ctc_loss:
                for k in ['rgb', 'keypoint', 'fuse']:
                    if f'{k}_gloss_probabilities_log' in head_outputs:
                        outputs[f'recognition_loss_{k}'] = self.compute_recognition_loss(
                        gloss_labels=gloss_labels, gloss_lengths=gls_lengths,
                        gloss_probabilities_log=head_outputs[f'{k}_gloss_probabilities_log'],
                        input_lengths=valid_len_out)
                    if self.cfg.get('handshape_heads', False) == True:
                        for hand in self.handshapes:
                            outputs[f'recognition_loss_{k}_handshape_{hand}'] = self.compute_recognition_loss(
                            gloss_labels=eval(f'handshape_labels_{hand}'), gloss_lengths=eval(f'hs_lengths_{hand}'),
                            gloss_probabilities_log=head_outputs[f'{k}_handshape_{hand}_probabilities_log'],
                            input_lengths=valid_len_out)
                for hand in self.handshapes:
                    if f'handshape_{hand}_probabilities_log' in head_outputs:      
                        outputs[f'recognition_loss_handshape_{hand}'] = self.compute_recognition_loss(
                            gloss_labels=eval(f'handshape_labels_{hand}'), gloss_lengths=eval(f'hs_lengths_{hand}'),
                            gloss_probabilities_log=head_outputs[f'handshape_{hand}_probabilities_log'],
                            input_lengths=valid_len_out)

            self.cfg['gloss_feature_ensemble'] = self.cfg.get('gloss_feature_ensemble','fuse_gloss_feature')
            if '@' in self.cfg['gloss_feature_ensemble']:
                feat_name, agg = self.cfg['gloss_feature_ensemble'].split('@')
                gloss_feature = [head_outputs[f'{k}_{feat_name}'] for k in ['fuse','rgb','keypoint','handshape']]
                if agg == 'cat':
                    gloss_feature = torch.cat(gloss_feature, dim=-1)
                elif agg == 'plus':
                    gloss_feature = sum(gloss_feature)
                else:
                    raise ValueError
                outputs['gloss_feature'] = gloss_feature
            else:
                stream, feat_name = self.cfg['gloss_feature_ensemble'].split('_gloss_')
                feat_name = 'gloss_'+feat_name
                outputs['gloss_feature'] = outputs[f'{stream}_{feat_name}']
            if compute_ctc_loss:
                outputs['recognition_loss'] = outputs['recognition_loss_rgb'] + outputs['recognition_loss_keypoint'] \
                    +outputs['recognition_loss_fuse']
                for hand in self.handshapes:
                    outputs['recognition_loss'] += outputs[f'recognition_loss_handshape_{hand}']
                if self.cfg.get('handshape_heads', False) == True:
                    for hand in self.handshapes:
                        outputs['recognition_loss'] += outputs[f'recognition_loss_rgb_handshape_{hand}'] + outputs[f'recognition_loss_keypoint_handshape_{hand}'] + \
                        outputs[f'recognition_loss_fuse_handshape_{hand}']
        else:
            raise ValueError
        
        if 'cross_entropy' in self.cfg:
            if ce_strategy == 'joint' or (ce_strategy == 'iterative' and step%2==1) or is_train==False:
                expand = self.cfg['cross_entropy'].get('expand', True)
                if type(self.cfg['cross_entropy']['types'])==list:
                    self.cfg['cross_entropy']['types']={t:self.cfg['cross_entropy'].get('loss_weight',1) 
                        for t in self.cfg['cross_entropy']['types']}
                for teaching_type, loss_weight in self.cfg['cross_entropy']['types'].items():
                    teacher = teaching_type.split('_teaches_')[0]
                    student = teaching_type.split('_teaches_')[1]
                    assert teacher in ['rgb', 'keypoint', 'ensemble_last','fuse','ensemble_early', 'handshape'] #, teacher#, 'fuse']
                    assert student == 'handshape'
                    pad_id = self.gloss_tokenizer.pad_id
                    blank_id = self.gloss_tokenizer.silence_id
                    if teacher != 'handshape':
                        ref_labels = gloss_labels
                        teacher_logits = outputs[f'{teacher}_gloss_logits'].detach()
                        gls_best_paths = best_ctc_path(teacher_logits, ref_labels, 
                                            valid_len_out, pad_id, blank_id, 
                                                expand=True)
                    for hand in self.handshapes:
                        if teacher == 'handshape':
                            teacher_logits = outputs[f'{teacher}_{hand}_logits'].detach()
                            ref_labels = eval(f'handshape_labels_{hand}')
                            best_paths = best_ctc_path(teacher_logits, ref_labels, 
                                            valid_len_out, pad_id, blank_id, 
                                            expand=True)
                        else:
                            hsih = eval(f'hs_seg_ids_{hand}')
                            gbp = gls_best_paths
                            hhl = outputs[f'handshape_{hand}_logits'].detach()
                            best_paths = best_ctc_gls_hs_path(gbp, hhl, hsih,
                                                    pad_id, blank_id, 
                                                    expand=expand)
                        teacher_labels = best_paths.detach() # B, T
                        student = f'handshape_{hand}'
                        student_logits = outputs[f'{student}_logits']
                        B, T, V = student_logits.shape
                        if expand:
                            loss_func = torch.nn.CrossEntropyLoss(reduction=self.reduction_method, ignore_index=self.gloss_tokenizer.pad_id) #divided by batch_size
                        else:
                            weights = torch.ones(V, dtype=torch.float, requires_grad=False) * loss_weight
                            weights[blank_id] = 0.1 * loss_weight
                            weights = weights.to(teacher_labels.device)
                            loss_func = torch.nn.CrossEntropyLoss(reduction=self.reduction_method, ignore_index=self.gloss_tokenizer.pad_id, 
                                                                weight=weights)
                        
                        outputs[f'{teaching_type}_{hand}_loss'] = loss_func(input=student_logits.view(-1, V), target=teacher_labels.view(-1).long())
                        outputs[f'{teaching_type}_{hand}_loss'] /=B
                        if 'recognition_loss' not in outputs:
                            outputs['recognition_loss'] = outputs[f'{teaching_type}_{hand}_loss']*loss_weight
                        else:  
                            oprs =  outputs['recognition_loss'] + outputs[f'{teaching_type}_{hand}_loss']*loss_weight
                            outputs['recognition_loss'] = oprs
        return outputs