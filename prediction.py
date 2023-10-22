import warnings, wandb
import pickle
from collections import defaultdict
from models.model import build_model
warnings.filterwarnings("ignore")
import argparse
import os, sys
sys.path.append(os.getcwd())#slt dir
import torch
from utils.misc import (
    get_logger,
    load_config,
    make_logger, 
    move_to_device,
    neq_load_customized
)
from dataset.Dataloader import build_dataloader
from utils.progressbar import ProgressBar
from utils.metrics import wer_list
from utils.phoenix_cleanup import clean_phoenix_2014_trans, clean_phoenix_2014

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def evaluation(model, val_dataloader, cfg, 
        tb_writer=None, wandb_run=None,
        epoch=None, global_step=None,
        generate_cfg={}, save_dir=None,
        do_recognition=True):  
    logger = get_logger()
    logger.info(generate_cfg)
    handshapes = cfg['model']['RecognitionNetwork'].get('handshape', [])
    print()
    if os.environ.get('enable_pbar', '1')=='1':
        pbar = ProgressBar(n_total=len(val_dataloader), desc='Validation')
    else:
        pbar = None
    if epoch!=None:
        logger.info('Evaluation epoch={} validation examples #={}'.format(epoch, len(val_dataloader.dataset)))
    elif global_step!=None:
        logger.info('Evaluation global step={} validation examples #={}'.format(global_step, len(val_dataloader.dataset)))
    model.eval()
    total_val_loss = defaultdict(int)
    results = defaultdict(dict)
    with torch.no_grad():
        for step, batch in enumerate(val_dataloader):
            #forward -- loss
            batch = move_to_device(batch, cfg['device'])
            forward_output = model(is_train=False, **batch)
            for k,v in forward_output.items():
                if '_loss' in k:
                    if type(v)!=int and v.dim()!=0 and 'contrastive' in cfg:
                        if cfg['contrastive'].get('negative_bp', False):
                            v = v.sum()
                        else:
                            v = v[0].sum()
                    total_val_loss[k] += v.item()
            if do_recognition: #wer
                #rgb/keypoint/fuse/ensemble_last_logits
                for k, gls_logits in forward_output.items():
                    if not 'gloss_logits' in k or gls_logits==None:
                        continue
                    logits_name = k.replace('gloss_logits','')
                    if logits_name in ['rgb_','keypoint_','fuse_','ensemble_last_','ensemble_early_','']:
                        if logits_name=='ensemble_early_':
                            input_lengths = forward_output['aux_lengths']['rgb'][-1]
                        else:
                            input_lengths = forward_output['input_lengths']
                        ctc_decode_output = model.predict_gloss_from_logits(
                            gloss_logits=gls_logits, 
                            beam_size=generate_cfg['recognition']['beam_size'], 
                            input_lengths=input_lengths
                        )  
                        batch_pred_gls = model.gloss_tokenizer.convert_ids_to_tokens(ctc_decode_output)
                        for name, gls_hyp, gls_ref in zip(batch['name'], batch_pred_gls, batch['gloss']):
                            results[name][f'{logits_name}gls_hyp'] = \
                                ' '.join(gls_hyp).upper() if model.gloss_tokenizer.lower_case \
                                    else ' '.join(gls_hyp)
                            results[name]['gls_ref'] = gls_ref.upper() if model.gloss_tokenizer.lower_case \
                                    else gls_ref 
                            # print(logits_name)
                            # print(results[name][f'{logits_name}gls_hyp'])
                            # print(results[name]['gls_ref'])
                    else:
                        print(logits_name)
                        raise ValueError
                if cfg['model']['RecognitionNetwork'].get('handshape_heads', False) == True or \
                    'handshape' in cfg['data']['input_streams']:
                    for hand in handshapes:
                        for k, handshape_logits in forward_output.items():
                            if not f'handshape_{hand}_logits' in k or handshape_logits==None:
                                continue
                            logits_name = k.replace(f'handshape_{hand}_logits', '')
                            if logits_name in ['rgb_','keypoint_','fuse_','ensemble_last_','ensemble_early_','']:
                                if logits_name=='ensemble_early_':
                                    input_lengths = forward_output['aux_lengths']['rgb'][-1]
                                else:
                                    input_lengths = forward_output['input_lengths']
                                ctc_decode_output = model.predict_gloss_from_logits(
                                    gloss_logits=handshape_logits, 
                                    beam_size=generate_cfg['recognition']['beam_size'], 
                                    input_lengths=input_lengths
                                )
                                batch_pred_hs = \
                                    eval(f"model.handshape_tokenizer_{hand}").convert_ids_to_tokens(ctc_decode_output)
                                for name, hs_hyp, hs_ref in \
                                    zip(batch['name'], batch_pred_hs, batch[f'handshape-{hand}']):
                                    results[name][f'{logits_name}hs_{hand}_hyp'] = ' '.join(hs_hyp)
                                    results[name][f'hs_{hand}_ref'] = " ".join([" ".join(hs) for hs in hs_ref])
                #multi-head
                if 'aux_logits' in forward_output:
                    for stream, logits_list in forward_output['aux_logits'].items(): #['rgb', 'keypoint]
                        lengths_list = forward_output['aux_lengths'][stream] #might be empty
                        for i, (logits, lengths) in enumerate(zip(logits_list, lengths_list)):
                            ctc_decode_output = model.predict_gloss_from_logits(
                                gloss_logits=logits, 
                                beam_size=generate_cfg['recognition']['beam_size'], 
                                input_lengths=lengths)
                            batch_pred_gls = model.gloss_tokenizer.convert_ids_to_tokens(ctc_decode_output)       
                            for name, gls_hyp, gls_ref in zip(batch['name'], batch_pred_gls, batch['gloss']):
                                results[name][f'{stream}_aux_{i}_gls_hyp'] = \
                                    ' '.join(gls_hyp).upper() if model.gloss_tokenizer.lower_case \
                                        else ' '.join(gls_hyp)      
            #misc
            if pbar:
                pbar(step)
        print()
    #logging and tb_writer
    for k, v in total_val_loss.items():
        logger.info('{} Average:{:.2f}'.format(k, v/len(val_dataloader)))
        if tb_writer:
            tb_writer.add_scalar('eval/'+k, v/len(val_dataloader), epoch if epoch!=None else global_step)
        if wandb_run:
            wandb.log({f'eval/{k}': v/len(val_dataloader)})
    #evaluation (Recognition:WER,  Translation:B/M)
    evaluation_results = {}
    if do_recognition:
        evaluation_results['wer'] = 200
        for hyp_name in results[name].keys():
            if not 'gls_hyp' in hyp_name:
                continue
            k = hyp_name.replace('gls_hyp','')
            if cfg['data']['dataset_name'].lower()=='phoenix-2014t':
                gls_ref = [clean_phoenix_2014_trans(results[n]['gls_ref']) for n in results]
                gls_hyp = [clean_phoenix_2014_trans(results[n][hyp_name]) for n in results] 
            elif cfg['data']['dataset_name'].lower()=='phoenix-2014':
                gls_ref = [clean_phoenix_2014(results[n]['gls_ref']) for n in results]
                gls_hyp = [clean_phoenix_2014(results[n][hyp_name]) for n in results] 
            elif cfg['data']['dataset_name'].lower() in ['csl-daily','cslr']:
                gls_ref = [results[n]['gls_ref'] for n in results]
                gls_hyp = [results[n][hyp_name] for n in results]
            wer_results = wer_list(hypotheses=gls_hyp, references=gls_ref)
            evaluation_results[k+'wer_list'] = wer_results
            logger.info('{}WER: {:.2f}'.format(k,wer_results['wer']))

            if tb_writer:
                tb_writer.add_scalar(f'eval/{k}WER', wer_results['wer'], epoch if epoch!=None else global_step)   
            if wandb_run!=None:
                wandb.log({f'eval/{k}WER': wer_results['wer']})
            evaluation_results['wer'] = min(wer_results['wer'], evaluation_results['wer'])
        
        for hand in handshapes:
            evaluation_results[f'wer_{hand}'] = 200
            for hyp_name in results[name].keys():
                if not f'hs_{hand}_hyp' in hyp_name:
                    continue
                k = hyp_name.replace('hyp','')
                hs_ref = [results[n][f'hs_{hand}_ref'] for n in results]
                hs_hyp = [results[n][hyp_name] for n in results]
                wer_results = wer_list(hypotheses=hs_hyp, references=hs_ref)
                evaluation_results[k+'wer_list'] = wer_results
                logger.info('{}WER: {:.2f}'.format(k,wer_results['wer']))

                if tb_writer:
                    tb_writer.add_scalar(f'eval/{k}WER', wer_results['wer'], epoch if epoch!=None else global_step)   
                if wandb_run!=None:
                    wandb.log({f'eval/{k}WER': wer_results['wer']})
                if cfg['model']['RecognitionNetwork']['fuse_method'] == 'empty':
                    evaluation_results[f'wer_{hand}'] = min(wer_results['wer'], evaluation_results[f'wer_{hand}'])
    #save
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'results.pkl'),'wb') as f:
            pickle.dump(results, f)
        with open(os.path.join(save_dir, 'evaluation_results.pkl'),'wb') as f:
            pickle.dump(evaluation_results, f)
    return evaluation_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser("SLT baseline Testing")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        type=str,
        help="Training configuration file (yaml).",
    )
    parser.add_argument(
        "--save_subdir",
        default='prediction',
        type=str
    )
    parser.add_argument(
        '--ckpt_name',
        default='best.ckpt',
        type=str
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    model_dir = cfg['training']['model_dir']
    os.makedirs(model_dir, exist_ok=True)
    global logger
    logger = make_logger(model_dir=model_dir, log_file='prediction.log')
    cfg['device'] = torch.device('cuda')
    model = build_model(cfg)
    #load model
    load_model_path = os.path.join(model_dir,'ckpts',args.ckpt_name)
    if os.path.isfile(load_model_path):
        state_dict = torch.load(load_model_path, map_location='cuda')
        neq_load_customized(model, state_dict['model_state'], verbose=True)
        epoch, global_step = state_dict.get('epoch',0), state_dict.get('global_step',0)
        logger.info('Load model ckpt from '+load_model_path)
    else:
        logger.info(f'{load_model_path} does not exist')
        epoch, global_step = 0, 0
    for split in ['dev','test']:
        logger.info('Evaluate on {} set'.format(split))
        dataloader, sampler = build_dataloader(cfg, split, model.gloss_tokenizer, 
            model.handshape_tokenizer_right, model.handshape_tokenizer_left)
        evaluation(model=model, val_dataloader=dataloader, cfg=cfg, 
                epoch=epoch, global_step=global_step, 
                generate_cfg=cfg['testing']['cfg'],
                save_dir=os.path.join(model_dir,args.save_subdir,split),
                do_recognition=True)

