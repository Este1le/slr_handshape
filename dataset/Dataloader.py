from dataset.VideoLoader import load_batch_video
from dataset.FeatureLoader import load_batch_feature
from dataset.Dataset import build_dataset
import torch
import pickle


def collate_fn_(inputs, cfg, data_cfg, is_train, 
    gloss_tokenizer=None,
    handshape_tokenizer_right=None, handshape_tokenizer_left=None,
    name2keypoint=None, signer2name_dict=None):
    outputs = {
        'name':[i['name'] for i in inputs],
        'gloss':[i.get('gloss','') for i in inputs],
        'text':[i.get('text','') for i in inputs],
        'alignment':[i.get('alignment', None) for i in inputs],
        'num_frames':[i['num_frames'] for i in inputs],
        'signers':[i['signer'] for i in inputs],
        'handshape-right': [i.get('handshape-right', None) for i in inputs],
        'handshape-left': [i.get('handshape-left', None) for i in inputs],
        }
    sgn_videos, sgn_keypoints, hs_videos, sgn_lengths = load_batch_video(
    data_cfg=data_cfg, 
    names=outputs['name'], 
    num_frames=outputs['num_frames'],
    transform_cfg=data_cfg['transform_cfg'],  
    dataset_name=data_cfg['dataset_name'], 
    is_train=is_train,  
    name2keypoint=name2keypoint
    )
    #print("output[gloss]:", outputs['gloss'])
    #print("output[num_frames]:", outputs['num_frames'])
    outputs['recognition_inputs'] = gloss_tokenizer(outputs['gloss'])
    outputs['recognition_inputs']['sgn_videos'] = sgn_videos
    outputs['recognition_inputs']['sgn_keypoints'] = sgn_keypoints
    outputs['recognition_inputs']['hs_videos'] = hs_videos
    outputs['recognition_inputs']['sgn_lengths'] = sgn_lengths
    for hand in ['right', 'left']:
        if eval(f"handshape_tokenizer_{hand}") != None and not all(e is None for e in outputs['handshape-'+hand]):
            outputs['recognition_inputs'].update(eval(f"handshape_tokenizer_{hand}")(outputs['handshape-{}'.format(hand)]))
    return outputs

def build_dataloader(cfg, split, 
    gloss_tokenizer=None, 
    handshape_tokenizer_right=None, handshape_tokenizer_left=None,
    mode='auto', val_distributed=False):
    dataset = build_dataset(cfg['data'], split)
    mode = split if mode=='auto' else mode
    if mode=='train':
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, 
            shuffle=cfg['training']['shuffle'] and split=='train'
        )
    else:
        if val_distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)
    batch_size = cfg['training']['batch_size'] 
    signer2name_dict = None
    dataloader = torch.utils.data.DataLoader(dataset,
                                             collate_fn=lambda x:collate_fn_(
                                                 inputs=x,
                                                 cfg=cfg,
                                                 data_cfg=cfg['data'],
                                                 is_train=(mode=='train'),
                                                 gloss_tokenizer=gloss_tokenizer,
                                                 handshape_tokenizer_right=handshape_tokenizer_right,
                                                 handshape_tokenizer_left=handshape_tokenizer_left,
                                                 name2keypoint=dataset.name2keypoints,
                                                 signer2name_dict=signer2name_dict),
                                             batch_size=batch_size,
                                             num_workers=cfg['training'].get('num_workers',2),
                                             sampler=sampler,
                                             )
    return dataloader, sampler