"""Correctness + perf for paged/GQA blk64 on SM100."""
import sys, math, time, torch
sys.path.insert(0, ".")
sys.path.insert(0, "/sparse/flashinfer")
from bsa_blk64_paged import build_page_pool, attn_paged, PAGE

DEV = "cuda"
def cos(a,b):
    return torch.nn.functional.cosine_similarity(a.float().flatten().unsqueeze(0),
                                                 b.float().flatten().unsqueeze(0)).item()

def ref(q, k_paged, v_paged, q2k, cnts, hkv):
    B,S,HQ,D = q.shape
    out = torch.zeros(B,S,HQ,D, device=DEV, dtype=torch.float32)
    kflat = k_paged.permute(1,0,2,3).reshape(hkv, -1, D)      # [H_kv, pages*64, D]
    vflat = v_paged.permute(1,0,2,3).reshape(hkv, -1, D)
    g = HQ // hkv
    for h in range(HQ):
        kv = h // g
        for b in range(S // PAGE):
            n = int(cnts[0,h,b]); pages = q2k[0,h,b,:n].tolist()
            cols = torch.cat([torch.arange(p*PAGE,(p+1)*PAGE, device=DEV) for p in pages])
            qq = q[0, b*PAGE:(b+1)*PAGE, h, :].float()
            s = (qq @ kflat[kv][cols].float().T) * (D ** -0.5)
            out[0, b*PAGE:(b+1)*PAGE, h, :] = torch.softmax(s, -1) @ vflat[kv][cols].float()
    return out

def bench(fn, warmup=10, rep=30):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(rep):
        a,b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b))
    ts.sort(); return ts[len(ts)//2]

def run(S, HQ, HKV, TOPK, check):
    pages = S // PAGE
    q = torch.randn(1,S,HQ,128, device=DEV, dtype=torch.bfloat16)
    kp = torch.randn(pages,HKV,PAGE,128, device=DEV, dtype=torch.bfloat16)
    vp = torch.randn(pages,HKV,PAGE,128, device=DEV, dtype=torch.bfloat16)
    k_pool, v_pool = build_page_pool(kp, vp)
    nqb = S // PAGE
    idx = torch.zeros(1,HQ,nqb,TOPK, dtype=torch.int32, device=DEV)
    cnt = torch.zeros(1,HQ,nqb, dtype=torch.int32, device=DEV)
    for b in range(nqb):
        n = min(TOPK, b+1)                       # causal between pages
        idx[0,:,b,:n] = torch.arange(n, device=DEV, dtype=torch.int32)
        cnt[0,:,b] = n
    o = attn_paged(q, k_pool, v_pool, idx, num_heads_kv=HKV,
                   block_sparse_num=TOPK, block_nums=cnt)
    msg = ""
    if check:
        r = ref(q, kp, vp, idx, cnt, HKV)
        msg = f"  cos={cos(o, r):.6f}"
    t = bench(lambda: attn_paged(q, k_pool, v_pool, idx, num_heads_kv=HKV,
                                 block_sparse_num=TOPK, block_nums=cnt))
    print(f"S={S:>6} HQ/HKV={HQ}/{HKV} topk={TOPK} | {t:8.4f} ms{msg}", flush=True)
    del q,kp,vp,k_pool,v_pool,idx,cnt,o; torch.cuda.empty_cache()

print("=== correctness (small) ===")
run(512, 4, 1, 4, True)
run(512, 32, 4, 4, True)
print("=== perf, Compass-V4 shape ===")
for S in (32768, 65536):
    run(S, 32, 4, 16, False)
print("DONE")
