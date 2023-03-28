# python -m torch.distributed.run --nproc_per_node=1 --master_port='29501' train.py --cfg-path lavis/projects/blip2/train/pretrain_stage1.yaml
python -m torch.distributed.run --nproc_per_node=4 train.py --cfg-path lavis/projects/blip2/train/pretrain_stage1_med.yaml

## DeepSpeed
# python -m torch.distributed.run --nproc_per_node=4 train.py --cfg-path lavis/projects/blip2/train/pretrain_stage1_med.yaml --deepspeed lavis/projects/blip2/train/ds_config.json
# deepspeed --num_gpus=4 train.py --cfg-path lavis/projects/blip2/train/pretrain_stage1_med.yaml --deepspeed lavis/projects/blip2/train/ds_config.json