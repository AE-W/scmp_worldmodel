"""Leave-one-IN sensitivity shard (aligned with the repo's original
sensitivity_sweep.py method, extended to all 6 ops):

For target (op, block): ONLY `op` is SC-enabled, and its skip set = all blocks
EXCEPT `block` — i.e. exactly one operator instance runs on SC, everything
else is FP. Sensitivity of (op, block) = how much latent L2 rises vs the pure
FP reference. Network is ~all-FP so each sample is fast (~30-40s @PNDM50).

block = -1 with op = "fp" means the pure-FP reference run.

Per-sample result files (out_dir/loi_{op}_{block}/{key}.json) enable resume.

Usage:
  PYTHONPATH=. BRIDGE_ROOT=... CUDA_VISIBLE_DEVICES=N python3 evaluate/sensitivity_shard_loi.py \
    --config configs/evaluation/bridge/frame_ada_sc_full.yaml \
    --op mlp_fc2 --block 4 --keys_file results/diverse_300.json \
    --key_start 0 --key_end 100 --out_dir /edrive2/qiuyid/sens300_loi
"""
import argparse, json, os
import numpy as np, torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKL

from models.sc_integration import reconfigure, set_skip_blocks, clear_skip_blocks, get_config
from evaluate.eval_local_n_samples import (
    GT_LATENT_DIR, ANNOT_DIR, C_ACT_SCALER, SEQUENCE_LENGTH,
    build_args, load_model, make_pipe, compute_actions_for_slice,
)

ALL_OPS = ("qkv", "qk", "av", "proj", "mlp_fc1", "mlp_fc2")


def key_parts(k):
    parts = k.split("_")
    return "_".join(parts[:-2]), int(parts[-2]), int(parts[-1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--op", required=True, choices=list(ALL_OPS) + ["fp"])
    p.add_argument("--block", type=int, required=True)  # -1 + op=fp -> FP reference
    p.add_argument("--keys_file", required=True)
    p.add_argument("--key_start", type=int, default=0)
    p.add_argument("--key_end", type=int, default=None)
    p.add_argument("--inference_steps", type=int, default=50)
    p.add_argument("--scheduler", default="PNDM")
    p.add_argument("--depth", type=int, default=28)
    p.add_argument("--out_dir", required=True)
    cli = p.parse_args()

    all_keys = json.load(open(cli.keys_file))
    end = cli.key_end if cli.key_end is not None else len(all_keys)
    keys = all_keys[cli.key_start:end]

    tag = "fp_ref" if cli.op == "fp" else f"{cli.op}_{cli.block}"
    bdir = os.path.join(cli.out_dir, f"loi_{tag}")
    os.makedirs(bdir, exist_ok=True)
    todo = [k for k in keys if not os.path.exists(os.path.join(bdir, f"{k}.json"))]
    if not todo:
        print(f"[{tag}] keys[{cli.key_start}:{end}] ALL DONE", flush=True)
        return

    args = build_args(cli.config, cli.inference_steps)
    device = torch.device("cuda:0")
    vae = AutoencoderKL.from_pretrained(args.vae_model_path, subfolder="vae").to(device).eval()
    vae.requires_grad_(False)
    model = load_model(args, device)

    # leave-one-in: enable ONLY target op, and only at target block
    reconfigure(args.attention_mode)
    clear_skip_blocks()
    cfg = get_config()
    for f in ALL_OPS:
        setattr(cfg, f"enable_{f}", False)
    if cli.op != "fp":
        setattr(cfg, f"enable_{cli.op}", True)
        set_skip_blocks(cli.op, [b for b in range(cli.depth) if b != cli.block])

    pipe = make_pipe(args, vae, model, cli.scheduler)
    print(f"[LOI {tag}] todo {len(todo)}/{len(keys)} steps={cli.inference_steps} "
          f"sched={cli.scheduler}", flush=True)

    for k in todo:
        outp = os.path.join(bdir, f"{k}.json")
        if os.path.exists(outp):
            continue
        try:
            eid, cam, start = key_parts(k)
            ann_path = os.path.join(ANNOT_DIR, f"{eid}.json")
            if not os.path.exists(ann_path):
                json.dump({"key": k, "tag": tag, "l2": None, "err": "no_annotation"}, open(outp, "w"))
                continue
            gt_lat = torch.load(os.path.join(GT_LATENT_DIR, f"{k}.pt"), weights_only=False, map_location=device)
            ann = json.load(open(ann_path))
            if start + SEQUENCE_LENGTH > len(ann["state"]):
                json.dump({"key": k, "tag": tag, "l2": None, "err": "oob"}, open(outp, "w"))
                continue
            arm = np.array(ann["state"])[start:start + SEQUENCE_LENGTH, :6]
            grip = np.array(ann["continuous_gripper_state"])[start:start + SEQUENCE_LENGTH]
            actions = torch.from_numpy(
                compute_actions_for_slice(arm, grip) * C_ACT_SCALER).float().unsqueeze(0)
            with torch.no_grad():
                _, lat = pipe(
                    actions.to(device).float(), mask_x=gt_lat[0:1].unsqueeze(0).to(device).float(),
                    video_length=args.num_frames, height=args.video_size[0], width=args.video_size[1],
                    num_inference_steps=args.infer_num_sampling_steps, guidance_scale=args.guidance_scale,
                    device=device, return_dict=False, output_type="latent_only",
                )
            l2 = F.mse_loss(lat.squeeze(0)[1:], gt_lat[1:]).item()
            json.dump({"key": k, "tag": tag, "l2": round(l2, 6)}, open(outp, "w"))
            print(f"  [{tag}] {k} L2={l2:.4f}", flush=True)
        except Exception as e:
            print(f"  [{tag}] {k} FAILED: {e}", flush=True)
    print(f"[LOI {tag}] shard done", flush=True)


if __name__ == "__main__":
    main()
