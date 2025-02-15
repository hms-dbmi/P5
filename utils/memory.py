import torch
import torch.nn as nn
import torch.nn.init as torch_init
import random
import numpy as np
import random

class Memory(nn.Module):
    def __init__(self, dim,K):
        super().__init__()

        self.K=K

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.register_buffer("queue", torch.randn(dim, K))
        # self.queue = nn.functional.normalize(self.queue, dim=0)


    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        # assert self.K % batch_size == 0  # for simplicity

        if (ptr + batch_size  > self.K):
            ptr = self.K - batch_size

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer
        # print(keys.size())

        self.queue_ptr[0] = ptr



    @torch.no_grad()
    def _return_queue(self):
        batch_size = 32

        ids=random.sample(range(self.K),batch_size)


        # top_p = torch.index_select(self.queue, dim=1, index=ids)
        top_p=self.queue[:,ids]

        top_p=top_p.T
        # print(top_p.size())

        return top_p


if __name__ == '__main__':
    neg_tivate=Memory(768,65536)

    pos_mem=Memory(768,65536)

    aa=torch.randn((2,768))
    neg_tivate._dequeue_and_enqueue(aa)

    cc=neg_tivate._return_queue(aa)
    cc=nn.functional.normalize(cc, dim=1)
    print(cc.size())

    bb=torch.randn((2,768))
    pos_mem._dequeue_and_enqueue(bb)

    dd=pos_mem._return_queue(bb)

    dd=nn.functional.normalize(dd, dim=1)

    all_instances = torch.cat([bb, dd], dim=0)
    print(dd.size())




