#!/usr/bin/env python3
import argparse
import json
import shutil
import time
import uuid
import urllib.request
from pathlib import Path


NEGATIVE = (
    "photorealistic, hyperrealistic, realistic photo, smooth, smoothed geometry, "
    "gradient, blurry, soft edges, rounded edges, complex texture, ultra detailed, "
    "human, character, hand, fingers, full ship, boat, sail, mast, island, beach, "
    "sea, ocean, water, palm tree, terrain scene, diorama, miniature scene, "
    "environment, background scene, multiple objects, extra props, duplicate object, "
    "collage, grid, split screen, contact sheet, model sheet, turntable sheet, "
    "multiple views in one image, thumbnails, four objects, stand, pedestal, "
    "display base, loose accessories, cast shadow, text, logo, watermark"
)

STYLE = (
    "<s0><s1>, LOWPOLY, low-poly 3D asset, stylized game asset reference, "
    "flat shading, sharp polygon edges, angular geometry, simplified texture, "
    "clean polygon faces, single isolated asset only, exactly one asset, "
    "solid neutral gray background, centered, no ground base unless the asset is a tile, "
    "orthographic modeling reference, readable silhouette, no text, no watermark"
)


def api_json(base_url, path, payload=None, timeout=30):
    url = f"{base_url}{path}"
    if payload is None:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def build_workflow(args, positive, negative, seed, save_prefix):
    nodes = {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": args.steps,
                "cfg": args.cfg,
                "sampler_name": "dpmpp_2m_sde",
                "scheduler": "karras",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": args.ckpt}},
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": args.size, "height": args.size, "batch_size": 1},
        },
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["4", 1]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": save_prefix, "images": ["8", 0]}},
    }

    previous_model = ["4", 0]
    previous_clip = ["4", 1]
    for node_num, lora_spec in enumerate(args.lora, start=10):
        name, weight_text = lora_spec.split(":", 1)
        weight = float(weight_text)
        node_id = str(node_num)
        nodes[node_id] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": previous_model,
                "clip": previous_clip,
                "lora_name": name,
                "strength_model": weight,
                "strength_clip": weight,
            },
        }
        previous_model = [node_id, 0]
        previous_clip = [node_id, 1]

    nodes["3"]["inputs"]["model"] = previous_model
    nodes["6"]["inputs"]["clip"] = previous_clip
    nodes["7"]["inputs"]["clip"] = previous_clip
    return nodes


def wait_for_prompt(base_url, prompt_id, timeout):
    started = time.time()
    while time.time() - started < timeout:
        history = api_json(base_url, f"/history/{prompt_id}", timeout=30)
        if prompt_id in history:
            return history[prompt_id]
        time.sleep(1.5)
    raise TimeoutError(f"Timed out waiting for prompt {prompt_id}")


def positive_prompt(item):
    view = item["view"]
    if item.get("view_lock"):
        view_lock = item["view_lock"]
    elif view == "front":
        view_lock = "front orthographic view, straight-on modeling reference"
    elif view == "side":
        view_lock = "left side orthographic view, straight-on modeling reference"
    else:
        view_lock = "top-down orthographic view, camera directly above, visible footprint"

    return f"{STYLE}. OBJECT LOCK: {item['prompt']}. VIEW LOCK: {view_lock}."


def run_item(args, item, ordinal, total):
    asset_slug = item["asset_slug"]
    variation = int(item["variation"])
    view = item["view"]
    dest = args.output_root / asset_slug / f"v{variation}" / f"{view}.png"
    dest.parent.mkdir(parents=True, exist_ok=True)

    if args.resume and dest.exists():
        print(f"[{ordinal:03d}/{total}] skip existing {asset_slug} v{variation} {view}", flush=True)
        return {
            **item,
            "seed": args.seed_base + int(item["index"]),
            "file_path": str(dest),
            "status": "ok",
            "error": "",
            "skipped": True,
        }

    seed = args.seed_base + int(item["index"])
    save_prefix = f"{args.run_name}/{asset_slug}/{asset_slug}_v{variation}_{view}"
    workflow = build_workflow(args, positive_prompt(item), NEGATIVE, seed, save_prefix)

    try:
        queued = api_json(
            args.comfy_url,
            "/prompt",
            {"prompt": workflow, "client_id": str(uuid.uuid4())},
            timeout=60,
        )
        history = wait_for_prompt(args.comfy_url, queued["prompt_id"], args.timeout)
        status = history.get("status", {})
        if status.get("status_str") not in ("success", "completed"):
            raise RuntimeError(f"Comfy status: {status}")

        images = history.get("outputs", {}).get("9", {}).get("images", [])
        if not images:
            raise RuntimeError("SaveImage produced no image")

        image = images[0]
        source = args.comfy_output / image.get("subfolder", "") / image["filename"]
        if not source.exists():
            raise FileNotFoundError(str(source))
        shutil.copy2(source, dest)
        print(f"[{ordinal:03d}/{total}] ok {asset_slug} v{variation} {view}", flush=True)
        return {
            **item,
            "seed": seed,
            "file_path": str(dest),
            "status": "ok",
            "error": "",
            "skipped": False,
        }
    except Exception as exc:
        error_path = dest.with_suffix(".error.txt")
        error_path.write_text(str(exc), encoding="utf-8")
        print(f"[{ordinal:03d}/{total}] ERROR {asset_slug} v{variation} {view}: {exc}", flush=True)
        return {
            **item,
            "seed": seed,
            "file_path": str(dest),
            "status": "error",
            "error": str(exc),
            "skipped": False,
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--comfy-output", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    parser.add_argument("--ckpt", default="Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors")
    parser.add_argument("--lora", action="append", default=[])
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--cfg", type=float, default=4.7)
    parser.add_argument("--seed-base", type=int, default=3005150000)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    api_json(args.comfy_url, "/system_stats", timeout=30)
    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    args.output_root.mkdir(parents=True, exist_ok=True)

    manifest = []
    for ordinal, item in enumerate(prompts, start=1):
        manifest.append(run_item(args, item, ordinal, len(prompts)))

    manifest_path = args.output_root / f"manifest_{args.run_name}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    ok_count = sum(1 for item in manifest if item["status"] == "ok")
    print(f"DONE ok={ok_count} total={len(manifest)} manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
