In order to spawn up multiple processes per node, you can use either torch.distributed.launch or torch.multiprocessing.spawn.

# 一、环境变量法 environ.py

# 单节点
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29500 environ.py --dist-backend gloo ../../datasets/small/

# 多节点
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --master_addr=xxx --node_rank=0 --master_port=6666 environ.py ../../datasets/small/

python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --master_addr=xxx --node_rank=1 --master_port=6666 environ.py ../../datasets/small/





# 二、spawn派生法 spawn.py

# 单节点单gpu
python3 spawn.py  --dist-backend=gloo --node-rank=0 --gpu 0 -j 16 -b 512 ../../datasets/ImageNet/imagenet

# 单节点多gpu
python3 spawn.py  --dist-backend=gloo --node-rank=0  -j 16 -b 512 ../../datasets/ImageNet/imagenet

# 多节点
python3 spawn.py  --dist-backend=gloo --nnodes=2 --node-rank=0  -j 16 -b 512 ../../datasets/ImageNet/imagenet

python3 spawn.py  --dist-backend=gloo --nnodes=2 --node-rank=1  -j 16 -b 512 ../../datasets/ImageNet/imagenet


# 三、脚本派生法 multiproc.py

python multiproc.py --nproc_per_node=2 main.py  --dist-backend gloo ../../datasets/small
