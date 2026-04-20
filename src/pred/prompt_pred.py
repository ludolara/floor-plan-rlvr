import os
import re
import argparse

from dotenv import load_dotenv
from vllm import LLM, SamplingParams
from src.pred.feedback_generator import FeedbackGenerator
from src.pred.extract_output_json import extract_output_json
from src.plot.direct_visualizer import DirectVisualizer
from src.utils.create_example import build_prompt

load_dotenv()

MODEL_PATH = "/home/l/luislara/links/projects/aip-pal/luislara/output/rplan5_70B_r64_GRPO_7n/checkpoint-100"
USER_PROMPTS = ["A 8 room home.", "A 8 room home, with all the bedrooms in L-shape."]


def _select_least(candidates):
    """
    Select the best candidate by prioritizing:
    1. JSON validity (valid JSON first)
    2. Minimum total_overlap_area 
    """
    def _evaluate_candidate(candidate):
        # First priority: JSON validity
        try:
            output_json = extract_output_json(candidate.text)
            if not output_json:  # Invalid or empty JSON
                return (1, float('inf'))
            json_invalid = 0
        except Exception:
            return (1, float('inf'))  
        
        # Second priority: total overlap area
        try:
            analysis = FeedbackGenerator.analyze(output_json, {})
            overlap_area = analysis.get('total_overlap_area', float('inf'))
        except Exception:
            overlap_area = float('inf')
        
        return (json_invalid, overlap_area)
    
    return min(candidates, key=_evaluate_candidate)


def main(num_examples: int = 1) -> None:
    # os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ["VLLM_USE_V1"] = "0"

    if isinstance(USER_PROMPTS, str):
        prompts_list = [USER_PROMPTS]
    else:
        prompts_list = USER_PROMPTS

    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=4,
        # device="cuda",
    )
    sampling_params = SamplingParams(
        max_tokens=4096,
        temperature=0.7,
        top_p=0.9,
        n=10
    )

    output_dir = "results/prompt_results"
    os.makedirs(output_dir, exist_ok=True)
    
    for prompt_idx, user_prompt in enumerate(prompts_list):
        print(f"\n{'#'*60}")
        print(f"Processing prompt {prompt_idx + 1}/{len(prompts_list)}")
        print(f"{'#'*60}")
        
        formatted_prompt = build_prompt({"prompt": user_prompt})
        
        filename_base = re.sub(r'[^\w\s-]', '', user_prompt).strip()
        filename_base = re.sub(r'[-\s]+', '_', filename_base)
        filename_base = filename_base[:100]  # Limit length
        
        for i in range(num_examples):
            print(f"\n{'='*60}")
            print(f"Generating example {i+1}/{num_examples} for prompt {prompt_idx + 1}")
            print(f"{'='*60}")
            
            outputs = llm.generate([formatted_prompt], sampling_params)
            
            best_candidate = _select_least(outputs[0].outputs)
            generated_text = best_candidate.text.strip()

            print("Prompt:")
            print(user_prompt)
            print("\nModel output:")
            print(generated_text)
            
            output_json = extract_output_json(generated_text)
            if output_json:
                floorplan_data = output_json.copy()
                if 'rooms' in floorplan_data and 'spaces' not in floorplan_data:
                    floorplan_data['spaces'] = floorplan_data.pop('rooms')
                
                image_filename = f"{filename_base}_{prompt_idx + 1}_{i+1}.png"
                
                visualizer = DirectVisualizer()
                image_path = os.path.join(output_dir, image_filename)
                visualizer.save_visualization(floorplan_data, image_path)
                print(f"Floorplan image saved to: {image_path}")
            else:
                print(f"Warning: Could not extract valid JSON from output for prompt {prompt_idx + 1}, example {i+1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate floorplan from prompt")
    parser.add_argument(
        "--num_examples",
        type=int,
        default=1,
        help="Number of examples to generate (default: 1)"
    )
    args = parser.parse_args()
    main(num_examples=args.num_examples)
