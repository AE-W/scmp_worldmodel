"""SmoothQuant calibration for the SC'd linears (qkv / proj / fc1 / fc2).

Runs N calibration clips through the FP model (attention_mode from config but
all SC disabled), hooks every to-be-SC'd nn.Linear to accumulate per-channel
activation |max|, then computes smooth scales s = act^alpha / w^(1-alpha)
via scmp_kernels.quant and saves {module_name: (D,) tensor} to a .pt file.

At eval time pass SC_SMOOTH_SCALES=<file> (loader in eval_local_n_samples
attaches tensors as module._sc_smooth_scales; sc_linear_forward picks it up).

Usage:
  PYTHONPATH=. BRIDGE_ROOT=... CUDA_VISIBLE_DEVICES=N \
    python3 evaluate/calibrate_smoothquant.py \
      --config configs/evaluation/bridge/frame_ada_sc_full.yaml \
      --keys_file results/diverse_300.json --num_samples 8 --alpha 0.5 \
      --out results/smoothquant_scales.pt
"""
import argparse, json, os
import numpy as np, torch
from diffusers.models import AutoencoderKL

from scmp_kernels.quant import accumulate_act_scales, compute_smooth_scales
from models.sc_integration import reconfigure, get_config
from evaluate.eval_local_n_samples import (
    GT_LATENT_DIR, ANNOT_DIR, C_ACT_SCALER, SEQUENCE_LENGTH,
    build_args, load_model, make_pipe, compute_actions_for_slice,
)

TARGET_SUFFIXES = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--keys_file", required=True)
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--inference_steps", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--out", required=True)
    cli = p.parse_args()

    args = build_args(cli.config, cli.inference_steps)
    device = torch.device("cuda:0")
    vae = AutoencoderKL.from_pretrained(args.vae_model_path, subfolder="vae").to(device).eval()
    vae.requires_grad_(False)
    model = load_model(args, device)

    # calibration must see FP activations: disable every SC op
    reconfigure(args.attention_mode)
    cfg = get_config()
    for f in ("qkv", "qk", "av", "proj", "mlp_fc1", "mlp_fc2"):
        setattr(cfg, f"enable_{f}", False)

    # hook target linears: accumulate per-channel |max| of inputs
    act_scales, hooks = {}, []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and name.endswith(TARGET_SUFFIXES):
            def make_hook(n):
                def hook(m, inp, out):
                    act_scales[n] = accumulate_act_scales(
                        inp[0].float(), act_scales.get(n))
                return hook
            hooks.append(mod.register_forward_hook(make_hook(name)))
    print(f"hooked {len(hooks)} linears", flush=True)

    pipe = make_pipe(args, vae, model, "DPM")
    keys = json.load(open(cli.keys_file))[: cli.num_samples]
    for k in keys:
        parts = k.split("_"); eid = "_".join(parts[:-2]); start = int(parts[-1])
        ann = json.load(open(os.path.join(ANNOT_DIR, f"{eid}.json")))
        if start + SEQUENCE_LENGTH > len(ann["state"]):
            continue
        gt_lat = torch.load(os.path.join(GT_LATENT_DIR, f"{k}.pt"), weights_only=False, map_location=device)
        arm = np.array(ann["state"])[start:start + SEQUENCE_LENGTH, :6]
        grip = np.array(ann["continuous_gripper_state"])[start:start + SEQUENCE_LENGTH]
        actions = torch.from_numpy(compute_actions_for_slice(arm, grip) * C_ACT_SCALER).float().unsqueeze(0)
        with torch.no_grad():
            pipe(actions.to(device).float(), mask_x=gt_lat[0:1].unsqueeze(0).to(device).float(),
                 video_length=args.num_frames, height=args.video_size[0], width=args.video_size[1],
                 num_inference_steps=cli.inference_steps, guidance_scale=args.guidance_scale,
                 device=device, return_dict=False, output_type="latent_only")
        print(f"  calibrated on {k}", flush=True)
    for h in hooks:
        h.remove()

    scales = {}
    for name, mod in model.named_modules():
        if name in act_scales:
            scales[name] = compute_smooth_scales(
                act_scales[name].to(device), mod.weight.detach().float(), alpha=cli.alpha).cpu()
    os.makedirs(os.path.dirname(cli.out) or ".", exist_ok=True)
    torch.save({"alpha": cli.alpha, "scales": scales}, cli.out)
    print(f"saved {len(scales)} scale vectors -> {cli.out}", flush=True)


if __name__ == "__main__":
    main()
