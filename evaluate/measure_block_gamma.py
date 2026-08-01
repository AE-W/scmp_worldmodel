"""Per-(block, timestep) sensitivity via the 6Bit-Diffusion Gamma signal.

6Bit-Diffusion (arXiv:2603.18742) observes that a transformer block's relative
input-output change  Gamma = ||Y - X||_1 / ||X||_1  at step t-1 linearly
predicts the quantization error of that block's linears at step t. Gamma is
BLOCK-level and the step index is exact, so this signal needs none of the
row-ordering information whose absence killed row-level MP here (Spearman
0.02). This script records Gamma[t, b] over an FP teacher rollout; the
allocator then solves  min sum W(t,b) * sigma(L)^2  at matched average cycles.

    python evaluate/measure_block_gamma.py --out results/block_gamma.npz
"""
import argparse, json, os
import numpy as np
import torch

from evaluate.calibrate_mp_fractions import (
    build_args, load_model, make_pipe,
    GT_LATENT_DIR, ANNOT_DIR, C_ACT_SCALER, SEQUENCE_LENGTH,
    compute_actions_for_slice,
)
from models.sc_integration import reconfigure, get_config
from models.sc_integration.sc_controller import get_current_step


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_config", default="configs/evaluation/bridge/frame_ada_sc_full.yaml")
    p.add_argument("--keys_file", default="results/diverse_300.json")
    p.add_argument("--num_samples", type=int, default=2)
    p.add_argument("--inference_steps", type=int, default=50)
    p.add_argument("--out", required=True)
    cli = p.parse_args()

    args = build_args(cli.eval_config, cli.inference_steps)
    device = torch.device("cuda:0")
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(args.vae_model_path, subfolder="vae").to(device).eval()
    vae.requires_grad_(False)
    model = load_model(args, device)
    reconfigure(args.attention_mode)
    cfg = get_config()
    for f in ("qkv", "qk", "av", "qkv_proj", "proj", "mlp_fc1", "mlp_fc2"):
        if hasattr(cfg, f"enable_{f}"):
            setattr(cfg, f"enable_{f}", False)      # FP teacher

    blocks = list(model.blocks)
    n_steps = cli.inference_steps
    acc = np.zeros((n_steps, len(blocks)))
    cnt = np.zeros((n_steps, len(blocks)))

    hooks = []
    for bi, blk in enumerate(blocks):
        def mk(bi):
            def hook(_m, inp, out):
                x = inp[0].detach()
                y = out[0].detach() if isinstance(out, (tuple, list)) else out.detach()
                g = ((y - x).abs().sum() / x.abs().sum().clamp_min(1e-8)).item()
                step, _tot = get_current_step()
                if 0 <= step < n_steps:
                    acc[step, bi] += g
                    cnt[step, bi] += 1
            return hook
        hooks.append(blk.register_forward_hook(mk(bi)))

    pipe = make_pipe(args, vae, model, "PNDM")
    for key in json.load(open(cli.keys_file))[: cli.num_samples]:
        parts = key.split("_")
        eid, start = "_".join(parts[:-2]), int(parts[-1])
        ann = json.load(open(os.path.join(ANNOT_DIR, f"{eid}.json")))
        if start + SEQUENCE_LENGTH > len(ann["state"]):
            continue
        gt = torch.load(os.path.join(GT_LATENT_DIR, f"{key}.pt"),
                        weights_only=False, map_location=device)
        arm = np.array(ann["state"])[start:start + SEQUENCE_LENGTH, :6]
        grip = np.array(ann["continuous_gripper_state"])[start:start + SEQUENCE_LENGTH]
        act = torch.from_numpy(
            compute_actions_for_slice(arm, grip) * C_ACT_SCALER).float().unsqueeze(0)
        with torch.no_grad():
            pipe(act.to(device).float(), mask_x=gt[0:1].unsqueeze(0).to(device).float(),
                 video_length=args.num_frames, height=args.video_size[0],
                 width=args.video_size[1], num_inference_steps=cli.inference_steps,
                 guidance_scale=args.guidance_scale, device=device,
                 return_dict=False, output_type="latent_only")
        print(f"  teacher rollout {key}", flush=True)
    for h in hooks:
        h.remove()

    G = acc / np.maximum(cnt, 1)
    np.savez_compressed(cli.out, gamma=G)
    print(f"wrote {cli.out}: Gamma{G.shape}", flush=True)
    print("每步均值(前/中/后):",
          [round(float(G[i].mean()), 4) for i in (0, n_steps // 2, n_steps - 1)])
    print("每块均值范围:", round(float(G.mean(0).min()), 4), "~",
          round(float(G.mean(0).max()), 4))


if __name__ == "__main__":
    main()
