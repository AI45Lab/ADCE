#!/bin/bash

# Default parameters
dataset_name="2_digit_multiplication"
prompt="superhigh"
num_mask=2
interv_type="mask"
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
    --mask_fix_position)  # New parameter
      mask_fix_position="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Define the list of models
models=("llama-3-8b" "llama-3-70b")

# Iterate through the model list and run main.py
for model in "${models[@]}"; do
  echo "Running model: $model"
  python white_main.py --model_name "$model" \
                 --dataset_name "$dataset_name" \
                 --prompt "$prompt" \
                 --num_mask "$num_mask" \
                 --interv_type "$interv_type" \
                 --mask_fix_position "$mask_fix_position" 
  echo "Completed model: $model"
  echo "------------------------"
done
echo "All models completed"
