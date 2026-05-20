#!/usr/bin/env python3
from pathlib import Path

from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parents[1]

MODELS = [
    # (
    #     "fp5-llama3.3-rlvr",
    #     Path("/home/l/luislara/links/projects/aip-pal/luislara/output/rplan5_70B_r64_GRPO_7n/checkpoint-100"),
    #     5,
    #     "0.01 +/- 0.14",
    #     "8.96 +/- 0.00",
    # ),
    (
        "fp6-llama3.3-rlvr",
        Path("/home/l/luislara/links/projects/aip-pal/luislara/output/rplan6_70B_r64_GRPO_7n/checkpoint-100"),
        6,
        "0.02 +/- 0.17",
        "8.80 +/- 0.00",
    ),
    (
        "fp7-llama3.3-rlvr",
        Path("/home/l/luislara/links/projects/aip-pal/luislara/output/rplan7_70B_r64_GRPO_7n/checkpoint-100"),
        7,
        "0.10 +/- 0.40",
        "7.79 +/- 0.00",
    ),
    (
        "fp8-llama3.3-rlvr",
        Path("/home/l/luislara/links/projects/aip-pal/luislara/output/rplan8_70B_r64_GRPO_7n/checkpoint-100"),
        8,
        "0.15 +/- 0.48",
        "6.96 +/- 0.00",
    ),
]


def read_token() -> str:
    env_path = ROOT / ".env"
    for line in env_path.read_text(errors="ignore").splitlines():
        stripped = line.strip()
        if stripped.startswith("HF_TOKEN_WRITE="):
            token = stripped.split("=", 1)[1].strip().strip('"').strip("'")
            if token:
                return token
    raise RuntimeError("HF_TOKEN_WRITE was not found in .env")


def card(repo_name: str, room_count: int, compatibility: str, diversity: str) -> str:
    return f"""---
base_model: meta-llama/Llama-3.3-70B-Instruct
library_name: transformers
tags:
- floor-plan-generation
- rplan
- llama-3.3
- rlvr
- grpo
---

# {repo_name}

RLVR-trained Llama-3.3-70B model for generating {room_count}-room RPLAN floor plans from structured room and adjacency specifications.

## Model Details

- Base model: `meta-llama/Llama-3.3-70B-Instruct`
- Training stages: supervised fine-tuning followed by RLVR/GRPO
- Task: RPLAN {room_count}-room floor-plan generation
- Format: full merged causal language model checkpoint

## Reported RLVR Metrics

| Metric | Value |
|---|---:|
| Compatibility | {compatibility} |
| Diversity | {diversity} |

## Intended Use

Use this model to generate {room_count}-room residential floor plans in the JSON format used by the associated RPLAN experiments. It is intended for research and evaluation on RPLAN-style floor-plan generation.

## Limitations

The model is specialized to the {room_count}-room RPLAN task and may not generalize to other datasets, room-count regimes, architectural standards, or safety-critical design workflows.
"""


def main() -> None:
    token = read_token()
    api = HfApi(token=token)
    user = api.whoami(token=token)["name"]
    print(f"Authenticated as: {user}", flush=True)

    for repo_name, checkpoint, room_count, compatibility, diversity in MODELS:
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)

        repo_id = f"{user}/{repo_name}"
        print(f"Preparing {repo_id}", flush=True)
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=False,
            exist_ok=True,
            token=token,
        )

        print(f"Uploading README for {repo_id}", flush=True)
        api.upload_file(
            path_or_fileobj=card(repo_name, room_count, compatibility, diversity).encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
            token=token,
            commit_message="Add RLVR model card",
        )

        print(f"Uploading checkpoint folder for {repo_id}: {checkpoint}", flush=True)
        api.upload_folder(
            folder_path=str(checkpoint),
            repo_id=repo_id,
            repo_type="model",
            token=token,
            commit_message="Upload RLVR full model checkpoint",
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "model*.safetensors",
                "model.safetensors.index.json",
                "special_tokens_map.json",
                "tokenizer*",
                "trainer_state.json",
                "training_args.bin",
            ],
        )
        print(f"Done: https://huggingface.co/{repo_id}", flush=True)

    print("All RLVR models uploaded.", flush=True)


if __name__ == "__main__":
    main()
