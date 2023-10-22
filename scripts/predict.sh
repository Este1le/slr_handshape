conda activate slt
cd /exp/xzhang/slt/slr_handshape
config_file=best_model/config.yaml
python -m torch.distributed.launch \
    --nproc_per_node 1 \
    --master_port 29898 \
    --use_env prediction.py --config ${config_file}
