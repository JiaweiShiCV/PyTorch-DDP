import torch
import torch.distributed as dist

dist.init_process_group(backend="gloo",
                        init_method="file:///home/sjw/distributed_test",
                        world_size=1,
                        rank=0)
tensor_list = []
for dev_idx in range(torch.cuda.device_count()):
    tensor_list.append(torch.FloatTensor([1]).cuda(dev_idx))
print(tensor_list)
dist.all_reduce_multigpu(tensor_list)
