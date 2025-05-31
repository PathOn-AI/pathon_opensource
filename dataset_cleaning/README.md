# Clean and Combine LeRobot Datasets

Recording all episodes correctly during data collection is tedious and nearly impossible due to the difficulty of teleoperation. We provide a script that allows developers to select episodes from different datasets and combine them into a single LeRobot dataset.

**Please note:** Here we assume the episode duration is the same across all episodes from different datasets. If you want to filter and merge datasets with different episode durations, please submit a PR to change the script.

## Prerequisites

Ensure you have the required dependencies installed:
```bash
pip install lerobot torch tqdm argparse
```

Set your Hugging Face token for dataset access:
```bash
export HUGGINGFACE_HUB_TOKEN=your_huggingface_token_here
```

## Step 1: Generate judge.jsonl file and add your judgments

Create a judgment file where each episode is scored based on data quality:

- **0**: The episode cannot be used for training (failed execution, corrupted data, etc.)
- **1**: Might be useful for training (partially successful, minor issues)
- **2**: Good imitation learning data (successful execution, clean teleoperation)

```bash
python generate_judge.py --repo_ids "DanqingZ/so100_test_pick_green_4,DanqingZ/so100_test_pick_green_5,DanqingZ/so100_test_pick_green_6,DanqingZ/so100_test_pick_grey_1,DanqingZ/so100_test_pick_grey_2" --output_file "judge.jsonl"
```

This will generate a `judge.jsonl` file with entries like:
```json
{"repo_id": "DanqingZ/so100_test_pick_green_4", "episode": 0, "score": null}
{"repo_id": "DanqingZ/so100_test_pick_green_4", "episode": 1, "score": null}
```

**Manual Review Required:** Open the generated `judge.jsonl` file and manually assign scores (0, 1, or 2) to each episode based on your quality assessment. You can use the LeRobot dataset visualizer to help with this process.

## Step 2: Select high-quality episodes and combine datasets

Based on the judge.jsonl file, select episodes with a score of 2 (and optionally 1) and combine them into a single dataset:

```bash
python data_cleaning.py --repo_ids "DanqingZ/so100_test_pick_green_4,DanqingZ/so100_test_pick_green_5,DanqingZ/so100_test_pick_green_6,DanqingZ/so100_test_pick_grey_1,DanqingZ/so100_test_pick_grey_2" --judge_file "judge.jsonl" --hub_repo_id "DanqingZ/so100_filtered_pick_green_grey"
```

This script will:
- Read your quality judgments from `judge.jsonl`
- Filter episodes based on scores (typically score ≥ 2)
- Combine selected episodes into a new consolidated dataset
- Upload the cleaned dataset to the specified Hugging Face repository

**Visualization of the consolidated dataset:** https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2FDanqingZ%2Fso100_filtered_pick_green_grey%2Fepisode_1

## Step 3: Validate the output

Verify that episodes were correctly transferred by comparing specific episodes between the original and filtered datasets:

```bash
python validate.py --new_repo_id DanqingZ/so100_filtered_pick_green_grey --original_repo_id DanqingZ/so100_test_pick_grey_1 --new_episode 56 --original_episode 0
```

This validation script will:
- Load the specified episodes from both datasets
- Compare actions, observations, and states frame by frame
- Report whether the data matches exactly
- Display progress with a progress bar for large episodes



## Example Workflow

1. Collect data across multiple sessions → multiple repo IDs
2. Generate judge.jsonl for all repositories
3. Manually review and score each episode (0-2)
4. Filter and combine high-scoring episodes (score ≥ 2)
5. Validate random samples from the filtered dataset
6. Use the cleaned dataset for training

This process ensures you have a high-quality, consolidated dataset ready for imitation learning while maintaining data provenance and quality standards.