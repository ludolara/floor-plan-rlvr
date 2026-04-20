import argparse
import json
import random
from pathlib import Path


TEMPLATES = [
    (
        "Create a residential floor plan with exactly {room_count} rooms and a total area target of {total_area} m2.",
        "The space requirements are as follows: {space_constraints}",
        "The required adjacencies are: {adjacency_constraints} Make sure every one of these direct connections is present in the final layout.",
    ),
    (
        "Generate a home layout that contains {room_count} rooms and reaches {total_area} m2 of total area.",
        "Include the following spaces and size requirements: {space_constraints}",
        "Respect these direct connectivity constraints in the bubble diagram: {adjacency_constraints} All listed pairs should connect directly.",
    ),
    (
        "I need a floor plan with {room_count} rooms, and the overall design should total {total_area} m2.",
        "Use these room-level constraints when sizing the spaces: {space_constraints}",
        "Follow these mandatory adjacency relationships: {adjacency_constraints} Keep the connectivity consistent with these requirements.",
    ),
]


def _fmt_float(value):
    if isinstance(value, (int, float)):
        return f"{value:.1f}"
    return "unknown"


def _fmt_room_type(room_type):
    if not isinstance(room_type, str):
        return "unknown"
    return room_type.replace("_", " ")


def _join_phrases(parts):
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return f"{', '.join(parts[:-1])}, and {parts[-1]}"


def _space_constraints(spaces):
    phrases = []
    for space in spaces:
        if not isinstance(space, dict):
            continue

        room_id = space.get("id", "unknown")
        room_type = _fmt_room_type(space.get("room_type"))
        area = space.get("area")
        width = space.get("width")
        height = space.get("height")

        if isinstance(area, (int, float)):
            phrases.append(
                f"{room_id} ({room_type}) should have an area of {_fmt_float(area)} m2"
            )
            continue

        if isinstance(width, (int, float)) and isinstance(height, (int, float)):
            phrases.append(
                f"{room_id} ({room_type}) should measure {_fmt_float(width)} m by {_fmt_float(height)} m"
            )
            continue

        phrases.append(f"{room_id} ({room_type}) must be present")

    if not phrases:
        return "no explicit space constraints were provided."
    return _join_phrases(phrases) + "."


def _adjacency_constraints(input_graph):
    if not isinstance(input_graph, dict):
        return "no explicit adjacency constraints were provided."

    edges = set()
    for source, targets in input_graph.items():
        if not isinstance(source, str) or not isinstance(targets, list):
            continue
        for target in targets:
            if not isinstance(target, str) or source == target:
                continue
            edges.add(tuple(sorted((source, target))))

    if not edges:
        return "no explicit adjacency constraints were provided."

    relations = [f"between {a} and {b}" for a, b in sorted(edges)]
    return f"direct connections must exist {_join_phrases(relations)}."


def build_nl_prompt(prompt_data, rng):
    room_count = prompt_data.get("room_count", "unknown")
    total_area = _fmt_float(prompt_data.get("total_area"))
    spaces = prompt_data.get("spaces", [])
    input_graph = prompt_data.get("input_graph", {})

    p1, p2, p3 = rng.choice(TEMPLATES)
    paragraph_1 = p1.format(
        room_count=room_count,
        total_area=total_area,
    )
    paragraph_2 = p2.format(
        space_constraints=_space_constraints(spaces),
    )
    paragraph_3 = p3.format(
        adjacency_constraints=_adjacency_constraints(input_graph),
    )
    return "\n\n".join([paragraph_1, paragraph_2, paragraph_3]).strip()


def convert_all(base_dir, seed=None, overwrite=True):
    rng = random.Random(seed)
    prompt_files = sorted(base_dir.rglob("prompt.json"))

    converted = 0
    skipped = 0
    failed = 0

    for prompt_file in prompt_files:
        output_file = prompt_file.with_name("nl_prompt.json")

        if output_file.exists() and not overwrite:
            skipped += 1
            continue

        try:
            with prompt_file.open("r", encoding="utf-8") as f:
                prompt_data = json.load(f)

            nl_prompt = build_nl_prompt(prompt_data, rng)
            output_file.write_text(nl_prompt + "\n", encoding="utf-8")
            converted += 1
        except Exception:
            failed += 1

    return converted, skipped, failed, len(prompt_files)


def remove_zero_json_files(base_dir):
    removed = 0
    failed = 0

    for zero_json_file in sorted(base_dir.rglob("0.json")):
        try:
            zero_json_file.unlink()
            removed += 1
        except Exception:
            failed += 1

    return removed, failed


def main():
    parser = argparse.ArgumentParser(
        description="Convert each prompt.json into a plain-text natural-language nl_prompt.json."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("results/results8_GRPO_70B_natural_language"),
        help="Directory that contains folders with prompt.json files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for deterministic template selection.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite existing nl_prompt.json files.",
    )
    args = parser.parse_args()

    removed, remove_failed = remove_zero_json_files(args.base_dir)

    converted, skipped, failed, total = convert_all(
        base_dir=args.base_dir,
        seed=args.seed,
        overwrite=not args.no_overwrite,
    )

    print(
        f"Removed {removed} 0.json files ({remove_failed} failed removals). "
        f"Processed {total} prompt.json files. Converted: {converted}, skipped: {skipped}, failed: {failed}."
    )


if __name__ == "__main__":
    main()
