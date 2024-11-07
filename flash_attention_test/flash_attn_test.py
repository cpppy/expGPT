import torch
from flash_attn import flash_attn_func


def main():

    '''
    q: (batch_size, seqlen, nheads, headdim)
    k: (batch_size, seqlen, nheads_k, headdim)
    v: (batch_size, seqlen, nheads_k, headdim)
    '''

    batch_size = 2
    seq_len = 768
    nheads = 12
    headdim = 256

    q = torch.randn(batch_size, seq_len, nheads, headdim, dtype=torch.bfloat16).cuda()
    k = torch.randn(batch_size, seq_len, nheads, headdim, dtype=torch.bfloat16).cuda()
    v = torch.randn(batch_size, seq_len, nheads, headdim, dtype=torch.bfloat16).cuda()

    output = flash_attn_func(q, k, v)
    print(output.shape)

if __name__=='__main__':

    main()