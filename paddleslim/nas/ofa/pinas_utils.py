import paddle
import paddle.nn as nn

@paddle.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        paddle.ones_like(tensor)
        for _ in range(paddle.distributed.get_world_size())
    ]
    paddle.distributed.all_gather(tensors_gather, tensor)

    output = paddle.concat(tensors_gather, axis=0)
    return output

class MOCO(nn.Layer):
    def __init__(self, queue_len=40960, feat_dim=128,):
        super(MOCO, self).__init__()
        self.register_buffer("queue", paddle.randn((feat_dim, queue_len)))
        self.queue = nn.functional.normalize(self.queue, axis=0)
        self.register_buffer("queue_ptr", paddle.zeros(shape=(1,), dtype=paddle.int32))
        self.queue_len = queue_len
    
    @paddle.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = paddle.randperm(batch_size_all)

        # broadcast to all gpus
        paddle.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = paddle.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = paddle.distributed.get_rank()
        idx_this = idx_shuffle.reshape([num_gpus, -1])[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @paddle.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = paddle.distributed.get_rank()
        idx_this = idx_unshuffle.reshape([num_gpus, -1])[gpu_idx]

        return x_gather[idx_this]
    
    @paddle.no_grad()
    def _dequeue_and_enqueue(self, keys, _nranks=1):
        # gather keys before updating queue
        if _nranks > 1:
            keys = concat_all_gather(keys)
        # keys: [B, dim_f]
        # queue: [dim_f, queue_len]
        # print("[DEBUG]keys", keys.shape)
        # print('[DEBUG]queue', self.queue.shape)
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose([1, 0])
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue_ptr[0] = ptr
