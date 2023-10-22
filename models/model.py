import torch
from .recognition_multi_stream import RecognitionNetwork
from utils.misc import get_logger

class SignLanguageModel(torch.nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.logger = get_logger()
        self.device = cfg['device']
        model_cfg = cfg['model']
        self.frozen_modules = []
        self.recognition_network = RecognitionNetwork(
            cfg=model_cfg['RecognitionNetwork'],
            input_type = 'video',
            transform_cfg=cfg['data']['transform_cfg'],
            input_streams = cfg['data'].get('input_streams','rgb'))
        self.gloss_tokenizer = self.recognition_network.gloss_tokenizer
        self.handshape_tokenizer_right = self.recognition_network.handshape_tokenizer_right
        self.handshape_tokenizer_left = self.recognition_network.handshape_tokenizer_left

        if self.recognition_network.visual_backbone!=None:
            self.frozen_modules.extend(self.recognition_network.visual_backbone.get_frozen_layers())
        if self.recognition_network.visual_backbone_keypoint!=None:
            self.frozen_modules.extend(self.recognition_network.visual_backbone_keypoint.get_frozen_layers())
        if self.recognition_network.visual_backbone_handshape!=None:
            self.frozen_modules.extend(self.recognition_network.visual_backbone_handshape.get_frozen_layers())

    def forward(self, is_train, step=0, translation_inputs={}, recognition_inputs={}, **kwargs):
        model_outputs = self.recognition_network(is_train=is_train, step=step,
            **recognition_inputs)
        model_outputs['total_loss'] = model_outputs['recognition_loss']  if 'recognition_loss' in model_outputs else None          
        return model_outputs
    
    def predict_gloss_from_logits(self, gloss_logits, beam_size, input_lengths):
        return self.recognition_network.decode(
            gloss_logits=gloss_logits,
            beam_size=beam_size,
            input_lengths=input_lengths)

    def get_best_alignment(self, tf_gloss_logits, ref_labels, input_lengths, pad_id, blank_id, expand=True):
        return self.recognition_network.get_best_alignment(tf_gloss_logits, ref_labels, \
                                                           input_lengths, pad_id, blank_id, expand=expand)

    def set_train(self):
        self.train()
        for m in self.frozen_modules:
            m.eval()

    def set_eval(self):
        self.eval()

def build_model(cfg):
    model = SignLanguageModel(cfg)
    return model.to(cfg['device'])