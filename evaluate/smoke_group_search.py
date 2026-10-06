"""Verify search dispatch matches the deployed Linear path, including SQ."""
import torch
from scmp_kernels import sc_matmul
from scmp_kernels.mp import MPConfig
from evaluate.calibrate_group_mp import rung
from evaluate.group_mp_policy import mean_cycles, policies
from models.sc_integration import sc_linear as impl


def main():
    torch.manual_seed(11)
    levels = [128, 112, 96, 64, 32]
    impl._MP_CONFIG = MPConfig(stoc_len_levels=levels, level_fractions=[0, 0, 1, 0, 0])
    impl._MP_GROUP_CHUNK_D = 128
    impl._GRANULARITY = "per_row"
    impl._HALVE = impl._MP_FIXED_PREC = True
    for width in (1152, 4608):
        x = torch.randn(8, width, device="cuda")
        linear = torch.nn.Linear(width, 16, device="cuda").eval()
        linear._sc_smooth_scales = torch.rand(width, device="cuda") + .5
        metric = x.abs().view(8, -1, 128).amax(-1)
        samples = [policies(levels)[2], policies(levels)[8], policies(levels)[9]]
        for policy in samples:
            impl._MP_PER_MODULE = {("qkv", 0): {"fractions": policy["level_fractions"], "invert": policy["invert"]}}
            table = rung(metric, levels, policy)
            expected = sc_matmul(x, linear.weight.float(), granularity="per_row", mode="bipolar",
                                 sc_prec=8, stoc_len=128, chunk_d=128,
                                 halve_bipolar_stoc_len=True, smooth_scales=linear._sc_smooth_scales,
                                 rung_table=table, level_lens=levels) + linear.bias.float()
            actual = impl.sc_linear_forward(x, linear, op="qkv", block_idx=0)
            assert torch.equal(actual, expected), (actual - expected).abs().max().item()
            cost = torch.tensor(levels, device="cuda")[table.long()].float().mean().item()
            assert abs(cost - mean_cycles(table.numel(), levels, policy["level_fractions"])) < 1e-5
        uniform = sc_matmul(x, linear.weight.float(), granularity="per_row", mode="bipolar",
                            sc_prec=8, stoc_len=96, chunk_d=128,
                            halve_bipolar_stoc_len=True, smooth_scales=linear._sc_smooth_scales)
        ref = sc_matmul(x, linear.weight.float(), granularity="per_row", mode="bipolar",
                        sc_prec=8, stoc_len=128, chunk_d=128,
                        halve_bipolar_stoc_len=True, smooth_scales=linear._sc_smooth_scales,
                        rung_table=rung(metric, levels, samples[0]), level_lens=levels)
        assert torch.equal(uniform, ref), (uniform - ref).abs().max().item()
        print(f"width={width}: search/deployment and uniform identity passed", flush=True)


if __name__ == "__main__":
    main()
