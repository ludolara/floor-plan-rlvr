import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm
from vllm import LLM, SamplingParams

from src.dataset_convert.rplan_graph import RPLANGraph
from src.pred.extract_output_json import extract_output_json
from src.pred.feedback_generator import FeedbackGenerator
from src.utils.create_example import build_prompt

load_dotenv()

DEFAULT_MODEL_PATH = "/home/l/luislara/links/projects/aip-pal/luislara/output/rplan5_70B_r64_GRPO_7n/checkpoint-100"
DEFAULT_BASE_DIR = "results/results8_GRPO_70B_natural_language"


def _select_least(candidates, input_prompt):
    """
    Select the best candidate by prioritizing:
    1. JSON validity (valid JSON first)
    2. Minimum total_overlap_area
    3. Minimum compatibility_score
    """

    def _evaluate_candidate(candidate):
        # First priority: JSON validity
        try:
            output_json = extract_output_json(candidate.text)
            if not output_json:
                return (1, float("inf"), float("inf"))
            json_invalid = 0
        except Exception:
            return (1, float("inf"), float("inf"))

        # Second priority: total overlap area
        try:
            analysis = FeedbackGenerator.analyze(output_json, input_prompt)
            overlap_area = analysis.get("total_overlap_area", float("inf"))
        except Exception:
            overlap_area = float("inf")

        # Third priority: compatibility score
        try:
            output_graph = RPLANGraph.from_ds2d(output_json)
            expected_graph = RPLANGraph.from_labeled_adjacency(
                input_prompt.get("input_graph", {})
            )
            compatibility_score = output_graph.compatibility_score(expected_graph)
        except Exception:
            compatibility_score = float("inf")

        return (json_invalid, overlap_area, compatibility_score)

    return min(candidates, key=_evaluate_candidate)


def _sorted_prompt_files(base_dir: Path):
    def _sort_key(path: Path):
        parent = path.parent.name
        try:
            return (0, int(parent))
        except ValueError:
            return (1, parent)

    return sorted(base_dir.rglob("nl_prompt.json"), key=_sort_key)


def run_predictions(
    base_dir: Path,
    output_name: str,
    model_path: str,
    tensor_parallel_size: int,
    batch_size: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    num_candidates: int,
    overwrite: bool,
    limit: int | None,
):
    os.environ["VLLM_USE_V1"] = "0"

    prompt_files = _sorted_prompt_files(base_dir)
    if limit is not None:
        prompt_files = prompt_files[:limit]

    if not prompt_files:
        print(f"No nl_prompt.json files found in {base_dir}")
        return

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
    )

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        n=num_candidates,
    )

    converted = 0
    skipped = 0
    failed = 0

    for i in tqdm(range(0, len(prompt_files), batch_size), desc="Running NL inference"):
        batch_files = prompt_files[i : i + batch_size]

        current_batch = []
        for prompt_file in batch_files:
            output_file = prompt_file.with_name(output_name)
            if output_file.exists() and not overwrite:
                skipped += 1
                continue

            prompt_data_file = prompt_file.with_name("prompt.json")
            input_prompt = {}
            if prompt_data_file.exists():
                try:
                    with prompt_data_file.open("r", encoding="utf-8") as f:
                        input_prompt = json.load(f)
                except Exception:
                    input_prompt = {}

            current_batch.append((prompt_file, output_file, input_prompt))

        if not current_batch:
            continue

        batch_prompts = []
        for prompt_file, _, _ in current_batch:
            prompt_text = prompt_file.read_text(encoding="utf-8").strip()
            batch_prompts.append(build_prompt({"prompt": prompt_text}))

        outputs = llm.generate(batch_prompts, sampling_params)

        for (prompt_file, output_file, input_prompt), output in zip(current_batch, outputs):
            try:
                best_candidate = _select_least(output.outputs, input_prompt)
                output_json = extract_output_json(best_candidate.text.strip())

                with output_file.open("w", encoding="utf-8") as f:
                    json.dump(output_json, f, indent=4)
                    f.write("\n")

                converted += 1
            except Exception as exc:
                failed += 1
                print(f"Failed on {prompt_file}: {exc}")

    print(
        f"Processed {len(prompt_files)} nl prompts. "
        f"Converted: {converted}, skipped: {skipped}, failed: {failed}."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run model inference on nl_prompt.json files and save outputs as 0.json."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(DEFAULT_BASE_DIR),
        help="Directory containing folders with nl_prompt.json.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="0.json",
        help="Output file name written next to each nl_prompt.json.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Model checkpoint path for vLLM.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=4,
        help="Tensor parallel size for vLLM.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of prompts per generation batch.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum output tokens per prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter.",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=10,
        help="Number of candidates generated per prompt.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of nl_prompt.json files to process.",
    )

    args = parser.parse_args()

    run_predictions(
        base_dir=args.base_dir,
        output_name=args.output_name,
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        num_candidates=args.num_candidates,
        overwrite=args.overwrite,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
