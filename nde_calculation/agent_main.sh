#!/bin/bash

dataset_name="analytic_entailment"
prompt="superhigh"
num_mask=2
interv_type="rephrase"
mask_fix_position=0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset_name)
      dataset_name="$2"
      shift 2
      ;;
    --prompt)
      prompt="$2"
      shift 2
      ;;
    --num_mask)
      num_mask="$2"
      shift 2
      ;;
    --interv_type)
      interv_type="$2"
      shift 2
      ;;
    --mask_fix_position)
      mask_fix_position="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done



models=("claude")
versions=("claude-3-sonnet-20240229")

# Iterate through the model list and run main.py
for model in "${models[@]}"; do
  for version in "${versions[@]}"; do
    echo "Running model: $model, version: $version"
    python agent_main.py --model_name "$model" \
                  --version "$version" \
                  --dataset_name "$dataset_name" \
                  --prompt "$prompt" \
                  --num_mask "$num_mask" \
                  --interv_type "$interv_type" \
                  --mask_fix_position "$mask_fix_position" 
    echo "Completed model: $model"
    echo "------------------------"
  done
done
echo "All models completed"
