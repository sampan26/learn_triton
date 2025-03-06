import torch
import triton
import triton.language as tl
DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')

autotune_configs = [
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=4, num_warps=4),
    triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5, num_warps=2),
    triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE': 8}, num_stages=5, num_warps=2)
]
@triton.autotune(configs = autotune_configs, key=['M', 'N', 'K'])
@triton.jit
def _matmult_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_a_M, stride_a_K, 
    stride_b_K, stride_b_N, 
    stride_c_M, stride_c_N, 
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    PID = tl.program_id(axis=0)
    num_PID_along_M = tl.cdiv(M, BLOCK_SIZE_M)
    num_PID_along_N = tl.cdiv(N, BLOCK_SIZE_N)
    num_PID_in_group = GROUP_SIZE * num_PID_along_N

    group_id = PID // num_PID_in_group
    first_PID_in_group_along_M = group_id * GROUP_SIZE
    group_size_adj = min(num_PID_along_M - first_PID_in_group_along_M , GROUP_SIZE)

    PID_M = first_PID_in_group_along_M + ((PID % num_PID_in_group) % group_size_adj)
    PID_N = (PID % num_PID_in_group) // group_size_adj

    offset_M = PID_M * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offset_N = PID_N * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offset_K = tl.arange(0, BLOCK_SIZE_K)

    a_offsets = offset_M[:, None] * stride_a_M + offset_K[None, :] * stride_a_K
    b_offsets = offset_K[:, None] * stride_b_K + offset_N[None, :] * stride_b_N

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        mask = offset_K < K - k * BLOCK_SIZE_K

        a = tl.load(a_ptr + a_offsets, mask=mask[None, :], other=0.0)
        b = tl.load(b_ptr + b_offsets, mask=mask[:, None], other=0.0)

        accumulator = tl.dot(a, b, acc=accumulator)

        a_offsets += BLOCK_SIZE_K * stride_a_K
        b_offsets += BLOCK_SIZE_K * stride_b_K

    accumulator = accumulator.to(tl.float16)
    c_offsets = stride_c_M * offset_M[:, None] + stride_c_N * offset_N[None, :]
    c_mask = (offset_M[:, None] < M) & (offset_N[None, :] < N)
    tl.store(c_ptr + c_offsets, accumulator, mask=c_mask)

def matmul(a, b):
    assert a.ndim == b.ndim == 2, "only supports matrices, not vectors or tensors"
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    
    (M, K), (_, N) = a.shape, b.shape

    # allocates output
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']) * triton.cdiv(N, meta['BLOCK_SIZE_N']),
    ) 
    _matmult_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c

def test_matmul_kernel(size: tuple, atol=1e-2, rtol=1e-1, device=DEVICE): # TODO does rtol=0 mean we don't use rtol?

    torch.manual_seed(0)
    assert type(size) == tuple and len(size) == 2
    a = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
    b = torch.randn((512, 512), device=DEVICE, dtype=torch.float16)
    c_tri = matmul(a, b)
    c_ref = torch.matmul(a, b)
    torch.testing.assert_close(c_tri, c_ref, atol=atol, rtol=rtol)
    print("PASSED")

configs = [
    triton.testing.Benchmark(
        x_names = ["M", "N", "K"],
        x_vals = [128 * i for i in range(2, 33)],
        line_arg = "provider", 
        line_vals = ["torch", "triton"],
        line_names = ["PyTorch", "Triton"],
        styles = [("green", "-"), ("blue", "-")],
        ylabel = "TFLOPS", 
        plot_name = "matmul-performance",
        args={},
    )
]
@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)
    quantiles = [0.5, 0.05, 0.95]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
    perf = lambda ms: 3 * M * N * K * 1e-12 / (ms * 1e-3)

    return perf(ms), perf(max_ms), perf(min_ms)

if __name__ == "__main__":
    # always run unit-tests
    test_matmul_kernel(size=(1024, 1024))

    # Only run benchmark if explicitly requested
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark":
        benchmark.run(save_path='.', print_data=False)