# Causal-Mediation-Analysis-Framework
Core for the Paper: Beyond Surface Structure: A Causal Assessment of LLMs' Comprehension Ability


## Environment

```
OS: Ubuntu 22.04.2 LTS 

python=3.10.14
torch=2.4.1
transformers=4.43.4
llama-recipes=0.0.3
datasets=2.21.0
accelerate=0.34.2
evaluate=0.4.3
```


## Fine-tuning
``` bash
cd finetune

# civil comments
python finetune_civil.py --model_name llama-3-8b --level 0.3

# analytic entailment
python finetune_entail.py 
```


# ADCE & AICE Calculation
To test closed-source models, run commands below:
``` bash
cd dce_calculation

# civil comments
bash agent_main.sh --dataset_name 2_digit_multiplication --prompt superhigh  --interv_type mask --num_mask 2 --mask_fix_position 0 

# 2_digit_multiplication
bash agent_main.sh --dataset_name 2_digit_multiplication --prompt superhigh  --interv_type mask --num_mask 2 --mask_fix_position 0 

# analytic_entailment
bash agent_main.sh --dataset_name analytic_entailment --interv_type rephrase --prompt superhigh  --mask_fix_position 0 --num_mask 2 

# GSM8k
bash agent_main.sh --dataset_name GSM8k --prompt superhigh --interv_type mask  --mask_fix_position 0 --num_mask 2

# word_unscrambling
bash agent_main.sh --dataset_name word_unscrambling --mask_fix_position 2 --num_mask 1   --interv_type mask --prompt superhigh 

# CommonsenseQA
bash agent_main.sh --dataset_name  commonsenseqa --prompt csuperhigh --interv_type rephrase --num_mask 2 --mask_fix_position 0
```

To test open-source models, run commands below:
``` bash
cd dce_calculation

# civil comments
bash white_main.sh --dataset_name 2_digit_multiplication --prompt superhigh  --interv_type mask --num_mask 2 --mask_fix_position 0 

# 2_digit_multiplication
bash white_main.sh --dataset_name 2_digit_multiplication --prompt superhigh  --interv_type mask --num_mask 2 --mask_fix_position 0 

# analytic_entailment
bash white_main.sh --dataset_name analytic_entailment --interv_type rephrase --prompt superhigh --num_mask 2 --mask_fix_position 0

# GSM8k
bash white_main.sh --dataset_name GSM8k --prompt superhigh --interv_type mask --num_mask 2 --mask_fix_position 0 

# word_unscrambling
bash white_main.sh --dataset_name word_unscrambling --mask_fix_position 2 --num_mask 1   --interv_type mask --prompt superhigh 

# CommonsenQA
bash white_main.sh --dataset_name commonsenseqa --prompt csuperhigh --interv_type rephrase --num_mask 2 --mask_fix_position 0 
```

## Automatic paraphrase
```bash
cd intervention_rephrase
python generate_intervention_commonsenseqa.py
```
