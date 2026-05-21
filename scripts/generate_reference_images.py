#!/usr/bin/env python3
import argparse
import base64
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib import request, error

STYLE_PREFIX = (
    "Low-poly stylized pirate island Unity Asset Store pack style, flat shaded surfaces, "
    "clean geometry, simple colors, realistic proportions, production-friendly scale, "
    "neutral background, centered single object, no text, no watermark, orthographic clarity"
)

RED_FLAGS = [
    "perspective_view",
    "non_neutral_background",
    "extra_decorative_details",
    "multiple_objects",
    "text_or_logo",
]

LINE_RE = re.compile(r"- \*\*(Front|Side|Top) V([1-4]):\*\* `(.+)`")
ASSET_RE = re.compile(r"^##\s+(\d+)\.\s+(.+)$")


@dataclass
class PromptItem:
    asset_id: int
    asset_name: str
    asset_slug: str
    variation: int
    view: str
    prompt_text: str


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def parse_prompts(md_path: Path) -> List[PromptItem]:
    items: List[PromptItem] = []
    current_asset_id: Optional[int] = None
    current_asset_name: Optional[str] = None

    with md_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m_asset = ASSET_RE.match(line)
            if m_asset:
                current_asset_id = int(m_asset.group(1))
                current_asset_name = m_asset.group(2).strip()
                continue

            m_prompt = LINE_RE.match(line)
            if m_prompt and current_asset_id is not None and current_asset_name is not None:
                view = m_prompt.group(1).lower()
                variation = int(m_prompt.group(2))
                prompt_text = m_prompt.group(3).strip()
                items.append(
                    PromptItem(
                        asset_id=current_asset_id,
                        asset_name=current_asset_name,
                        asset_slug=slugify(current_asset_name),
                        variation=variation,
                        view=view,
                        prompt_text=prompt_text,
                    )
                )

    return items


def build_styled_prompt(raw_prompt: str, style_prefix: str) -> str:
    if style_prefix.lower() in raw_prompt.lower():
        return raw_prompt
    return f"{style_prefix}. {raw_prompt}"


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def output_path(root: Path, item: PromptItem) -> Path:
    return root / item.asset_slug / f"v{item.variation}" / f"{item.view}.png"


def manifest_headers() -> List[str]:
    return [
        "asset_id",
        "asset_name",
        "asset_slug",
        "variation",
        "view",
        "prompt_hash",
        "file_path",
        "status",
        "regen_count",
        "notes",
    ]


def load_manifest(manifest_path: Path) -> Dict[Tuple[str, int, str], Dict[str, str]]:
    if not manifest_path.exists():
        return {}
    rows: Dict[Tuple[str, int, str], Dict[str, str]] = {}
    with manifest_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["asset_slug"], int(row["variation"]), row["view"])
            rows[key] = row
    return rows


def save_manifest(manifest_path: Path, rows: List[Dict[str, str]]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_headers())
        writer.writeheader()
        writer.writerows(rows)


def pick_subset(items: List[PromptItem], mode: str) -> List[PromptItem]:
    if mode == "all":
        return items
    if mode == "pilot24":
        if len(items) < 24:
            raise ValueError("Not enough prompts for pilot24")
        return items[:24]
    raise ValueError(f"Unknown mode: {mode}")


def run_qa_heuristics(prompt: str) -> List[str]:
    prompt_l = prompt.lower()
    flags: List[str] = []
    if "orthographic" not in prompt_l:
        flags.append("perspective_view")
    if "neutral" not in prompt_l:
        flags.append("non_neutral_background")
    if "centered single" not in prompt_l and "single asset" not in prompt_l:
        flags.append("multiple_objects")
    if "no text" not in prompt_l or "no logo" not in prompt_l:
        flags.append("text_or_logo")
    return flags


def openai_image_generate(
    api_key: str,
    model: str,
    prompt: str,
    size: str,
    output_file: Path,
    retries: int,
) -> None:
    body = {
        "model": model,
        "prompt": prompt,
        "size": size,
    }
    url = "https://api.openai.com/v1/images/generations"
    data = json.dumps(body).encode("utf-8")

    for attempt in range(retries + 1):
        req = request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with request.urlopen(req, timeout=180) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            b64 = payload["data"][0].get("b64_json")
            if not b64:
                raise RuntimeError("Response did not contain b64_json image payload")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_bytes(base64.b64decode(b64))
            return
        except error.HTTPError as e:
            raw = e.read().decode("utf-8", errors="ignore")
            if attempt >= retries:
                raise RuntimeError(f"HTTPError {e.code}: {raw}") from e
            sleep_s = 2 ** attempt + random.random()
            time.sleep(sleep_s)
        except Exception:
            if attempt >= retries:
                raise
            sleep_s = 2 ** attempt + random.random()
            time.sleep(sleep_s)


def is_png(file_path: Path) -> bool:
    if not file_path.exists():
        return False
    sig = file_path.read_bytes()[:8]
    return sig == b"\x89PNG\r\n\x1a\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate reference images from markdown prompts")
    parser.add_argument("--md", default="references/asset_generation_prompts_756.md")
    parser.add_argument("--output-root", default="references/generated_756")
    parser.add_argument("--manifest", default="references/generated_756/manifest.csv")
    parser.add_argument("--mode", choices=["pilot24", "all"], default="pilot24")
    parser.add_argument("--size", default="768x768")
    parser.add_argument("--model", default="gpt-image-1")
    parser.add_argument("--style-prefix", default=STYLE_PREFIX)
    parser.add_argument("--qa-fail-threshold", type=float, default=0.10)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Regenerate even if file exists")
    args = parser.parse_args()

    md_path = Path(args.md)
    output_root = Path(args.output_root)
    manifest_path = Path(args.manifest)

    items = parse_prompts(md_path)
    if not items:
        print("No prompts parsed.", file=sys.stderr)
        return 1

    selected = pick_subset(items, args.mode)
    existing = load_manifest(manifest_path)

    rows: List[Dict[str, str]] = []
    qa_failures = 0

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not args.dry_run and not api_key:
        print("OPENAI_API_KEY is required unless --dry-run is used.", file=sys.stderr)
        return 2

    for item in selected:
        styled = build_styled_prompt(item.prompt_text, args.style_prefix)
        p_hash = prompt_hash(styled)
        out_path = output_path(output_root, item)
        key = (item.asset_slug, item.variation, item.view)
        prev = existing.get(key)

        status = "ok"
        regen_count = int(prev["regen_count"]) + 1 if prev and args.force else int(prev["regen_count"]) if prev else 0
        notes: List[str] = []

        flags = run_qa_heuristics(styled)
        if flags:
            notes.append("flags=" + "|".join(flags))
            qa_failures += 1

        needs_generate = args.force or (not out_path.exists())

        if args.dry_run:
            status = "planned"
            if flags:
                status = "qa_flagged"
        else:
            if needs_generate:
                try:
                    openai_image_generate(
                        api_key=api_key,
                        model=args.model,
                        prompt=styled,
                        size=args.size,
                        output_file=out_path,
                        retries=args.max_retries,
                    )
                except Exception as e:
                    status = "error"
                    notes.append(f"generation_error={e}")

            if status != "error":
                if not is_png(out_path):
                    status = "error"
                    notes.append("invalid_png")
                elif flags:
                    status = "qa_flagged"

        rows.append(
            {
                "asset_id": str(item.asset_id),
                "asset_name": item.asset_name,
                "asset_slug": item.asset_slug,
                "variation": str(item.variation),
                "view": item.view,
                "prompt_hash": p_hash,
                "file_path": str(out_path),
                "status": status,
                "regen_count": str(regen_count),
                "notes": ";".join(notes),
            }
        )

    save_manifest(manifest_path, rows)

    total = len(rows)
    fail_rate = qa_failures / total if total else 0.0
    print(f"Mode={args.mode} total={total} qa_failures={qa_failures} qa_fail_rate={fail_rate:.2%}")
    print(f"Manifest: {manifest_path}")

    if fail_rate > args.qa_fail_threshold:
        print(
            f"WARNING: qa_fail_rate {fail_rate:.2%} exceeds threshold {args.qa_fail_threshold:.2%}; "
            "update style_prefix and regenerate current batch.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
