CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python -m torch.distributed.run --master_port 29501 --nproc_per_node=7 train.py --cfg-path lavis/projects/blip2/train/pretrain_stage2_med.yaml