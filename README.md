Here is the full version, with your original paths preserved. 

````markdown
# Generative Floor Plan Design with LLMs via RLVR

[![arXiv](https://img.shields.io/badge/arXiv-2605.14117-b31b1b.svg)](https://arxiv.org/abs/2605.14117)
[![Hugging Face Collection](https://img.shields.io/badge/Hugging%20Face-Collection-FFD21E.svg)](https://huggingface.co/collections/ludolara/generative-floor-plan-design-with-llms-via-rlvr)

This repository contains the code for **Generative Floor Plan Design with LLMs via Reinforcement Learning with Verifiable Rewards**.

The project trains large language models to generate **2D floor plans as structured JSON** from explicit design constraints, including room count, target areas, and a bubble diagram specifying room connectivity. The full pipeline goes from RPLAN preprocessing to supervised fine-tuning, RLVR training, and batched inference with vLLM.

```text
RPLAN rasters -> HouseGAN++ JSONs -> Hugging Face datasets -> SFT -> RLVR -> vLLM inference
````

---

## Overview

Most prior floor-plan generation systems operate on raster images or condition only on room adjacency graphs. This repository instead uses a structured JSON-to-JSON formulation.

* **Input**: room count, total area, room-level constraints, and an input connectivity graph.
* **Output**: a complete floor plan with room polygons, room areas, room IDs, doors, and coordinates.
* **Training**: supervised fine-tuning followed by reinforcement learning with verifiable rewards.
* **Inference**: batched generation with vLLM and optional LoRA adapters.

The RLVR stage uses automatically computable rewards to encourage valid JSON outputs, correct connectivity, accurate total area, and non-overlapping room polygons.

---

## Repository Structure

```text
floor-plan-rlvr/
  datasets/
    rplan_json/

    rplan_5/
      train/
      validation/
      test/

    rplan_6/
      train/
      validation/
      test/

    rplan_7/
      train/
      validation/
      test/

    rplan_8/
      train/
      validation/
      test/

  scripts/
    train_multi_node.slurm
    grpo_multi_node.slurm

  src/
    dataset_convert/
      rplan.py

    pred/
      run_generation.py

    grpo/
      reward_calculator.py
```

---

## 1. Dataset Preparation

This repository expects RPLAN to be converted in two steps:

1. RPLAN raster annotations to HouseGAN++-style JSON.
2. HouseGAN++-style JSON to Hugging Face `DatasetDict`.

### Step 1: Request RPLAN

RPLAN is distributed through an official request process. Download it from the authors using the form below:

[https://docs.google.com/forms/d/e/1FAIpQLSfwteilXzURRKDI5QopWCyOGkeb_CFFbRwtQ0SOPhEg0KGSfw/viewform](https://docs.google.com/forms/d/e/1FAIpQLSfwteilXzURRKDI5QopWCyOGkeb_CFFbRwtQ0SOPhEg0KGSfw/viewform)

### Step 2: Convert RPLAN rasters to JSON

Use the HouseGAN data reader:

[https://github.com/sepidsh/Housegan-data-reader](https://github.com/sepidsh/Housegan-data-reader)

After conversion, place the resulting JSON files under:

```text
datasets/
  rplan_json/
```

### Step 3: Convert JSONs to Hugging Face datasets

Run:

```bash
python src/dataset_convert/rplan.py
```

This creates room-count-specific datasets:

```text
datasets/
  rplan_5/
    train/
    validation/
    test/

  rplan_6/
    train/
    validation/
    test/

  rplan_7/
    train/
    validation/
    test/

  rplan_8/
    train/
    validation/
    test/
```

Each dataset contains structured examples with:

* `room_count`
* `total_area`
* `spaces`
* room IDs and room types
* room-level area or dimension constraints
* an `input_graph` encoding the bubble diagram

---

## 2. Training

The training pipeline has two stages:

1. **Supervised Fine-Tuning (SFT)**: teaches the model to map structured JSON specifications to JSON-encoded floor plans.
2. **RLVR Fine-Tuning**: improves constraint satisfaction using automatically verifiable rewards.

---

### Stage 1: Supervised Fine-Tuning

The SFT stage teaches the model to translate structured JSON specifications into JSON-encoded floor plans.

Multi-node Slurm example:

```bash
sbatch scripts/train_multi_node.slurm 2 6
```

Arguments:

```text
EPOCHS ROOM_NUMBER
```

Example:

```text
2 6
```

This trains on the 6-room task for 2 epochs.

Before launching, adjust the following inside the Slurm script as needed:

* base model path
* dataset path
* output directory
* number of nodes
* number of GPUs
* LoRA configuration
* batch size
* context length

---

### Stage 2: RLVR Fine-Tuning

The RLVR stage improves constraint satisfaction using automatically verifiable rewards.

Run:

```bash
sbatch scripts/grpo_multi_node.slurm 6
```

Arguments:

```text
ROOM_NUMBER
```

Example:

```text
6
```

The RLVR rewards are based on:

* valid JSON parsing
* non-overlapping room polygons
* connectivity agreement with the input bubble diagram
* total area agreement with the requested floor-plan area

Invalid outputs, unparsable JSON, or layouts with overlapping room polygons receive zero reward.

Reward code is implemented in:

```text
src/grpo/reward_calculator.py
```

---

## 3. Inference with vLLM

Run batched generation with vLLM:

```bash
python src/pred/run_generation.py \
  --batch_size 64 \
  --model_name_or_path models/Llama-3.3-70B-Instruct \
  --lora_adapter_path output/sft_rplan6 \
  --dataset_name_or_path datasets/final_2/rplan_6 \
  --output_dir results/rplan6_demo \
  --test_range "1,100" \
  --use_sampling
```

The LoRA adapter is optional. To run directly from a merged model or base checkpoint, omit:

```bash
--lora_adapter_path output/sft_rplan6
```

Example without LoRA:

```bash
python src/pred/run_generation.py \
  --batch_size 64 \
  --model_name_or_path models/Llama-3.3-70B-Instruct \
  --dataset_name_or_path datasets/final_2/rplan_6 \
  --output_dir results/rplan6_demo \
  --test_range "1,100" \
  --use_sampling
```

---

## 4. Output Format

Inference results are written under:

```text
results/<experiment_name>/<sample_index>/
```

Each sample directory contains:

```text
0.json                  # Generated floor-plan prediction
prompt.json             # Input prompt/specification
analysis/sample.json    # Original dataset sample
```

The generated prediction is a JSON object containing an `output` field with:

* `room_count`
* `total_area`
* `spaces`
* room and door entries
* room IDs
* room types
* polygon coordinates
* computed areas

---

## 5. Evaluation Metrics

The paper reports both standard floor-plan generation metrics and additional constraint-adherence metrics.

### Standard metrics

* **Compatibility ↓**: graph edit distance between the requested bubble diagram and the reconstructed graph from the generated floor plan.
* **Realism ↑**: volunteer feedback comparing generated layouts against ground-truth layouts.
* **Diversity ↓**: FID between generated and ground-truth floor-plan renderings.

### Constraint-adherence metrics

* **Room Area ↓**: mean absolute percentage error of per-room area.
* **Room ID ↑**: exact-match accuracy of room IDs relative to the prompt.
* **Overlap ↓**: whether any generated room polygons overlap.
* **% Overlap ↓**: total overlapped area divided by generated total area.

---

## 6. Main Results

Using Llama-3.3-70B-Instruct with SFT + RLVR and best-of-10 inference, the model achieves strong constraint adherence across the 5-room to 8-room tasks.

|    Task | Compatibility ↓ | Diversity ↓ |
| ------: | --------------: | ----------: |
| 5 rooms |            0.01 |         9.0 |
| 6 rooms |            0.02 |         8.8 |
| 7 rooms |            0.10 |         7.8 |
| 8 rooms |            0.15 |         7.0 |

In the most complex 8-room setting, the method reduces Compatibility by 94% relative to HouseDiffusion, from 2.50 to 0.15, and improves Diversity by 26.32%, from 9.5 to 7.0.

---

## 7. Hugging Face Resources

Models and related resources are available in the Hugging Face collection:

[https://huggingface.co/collections/ludolara/generative-floor-plan-design-with-llms-via-rlvr](https://huggingface.co/collections/ludolara/generative-floor-plan-design-with-llms-via-rlvr)

---

## 8. Data Use Notice

This repository does not redistribute RPLAN or processed versions of RPLAN.

RPLAN is released under a restricted-access research-only data-use agreement. Users must request access through the official RPLAN form and follow the dataset terms, including restrictions on redistribution and commercial use.

---

## 9. Citation

If you use this repository, please cite:

```bibtex
@misc{lara2026fprlvr,
  title = {Generative Floor Plan Design with LLMs via Reinforcement Learning with Verifiable Rewards},
  author = {Lara, Luis and Milios, Aristides and Luo, Zhi Hao and Sharma, Aditya and Luo, Ge Ya and Beckham, Christopher and Golemo, Florian and Pal, Christopher},
  year = {2026},
  eprint = {2605.14117},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2605.14117}
}
```

---

## References

* RPLAN request form: [https://docs.google.com/forms/d/e/1FAIpQLSfwteilXzURRKDI5QopWCyOGkeb_CFFbRwtQ0SOPhEg0KGSfw/viewform](https://docs.google.com/forms/d/e/1FAIpQLSfwteilXzURRKDI5QopWCyOGkeb_CFFbRwtQ0SOPhEg0KGSfw/viewform)
* HouseGAN data reader: [https://github.com/sepidsh/Housegan-data-reader](https://github.com/sepidsh/Housegan-data-reader)
* Hugging Face collection: [https://huggingface.co/collections/ludolara/generative-floor-plan-design-with-llms-via-rlvr](https://huggingface.co/collections/ludolara/generative-floor-plan-design-with-llms-via-rlvr)

```
```
