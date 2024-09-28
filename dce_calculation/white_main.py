import subprocess
import datetime
import argparse
import os
import shutil

def run_script(script_name, args):
    command = ["python", script_name] + [f"--{k}={v}" for k, v in args.items() if v is not None]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error running {script_name}:")
        print(stderr.decode())
    return stdout.decode()

def main():
    parser = argparse.ArgumentParser(description="Run llm_first.py and llm_second_intervention_norephrase.py")
    
    # Add arguments for llm_first.py
    parser.add_argument('--dataset_name', type=str, default='analytic_entailment')
    parser.add_argument('--model_name', type=str, default='llama-3-8b')
    parser.add_argument('--prompt', type=str, default='superhigh')
    parser.add_argument('--mask_fix_position', type=int, default= 0)
     # Add arguments for llm_second_intervention_norephrase.py
    parser.add_argument('--num_mask', type=int, default=2)
    parser.add_argument('--interv_type', type=str, default='rephrase')
    parser.add_argument('--log_file_path', type=str, default=None)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--max_new_tokens', type=int, default=20)
    parser.add_argument('--per_num_mask', type=int, default=1)
    
    
    args = parser.parse_args()

    if args.log_file_path is None:
    # Generate log file path
        now = datetime.datetime.now()
        current_time = now.strftime("%m%d_%H%M")
        args.log_file_path = f"acc_{args.model_name}_{current_time}"
        folder_path = f"./results/{args.dataset_name}/results_{args.prompt[0]}/{args.log_file_path}/"
        print('folder_path:',folder_path)

        # Ensure the folder exists
        os.makedirs(os.path.dirname(folder_path), exist_ok=True)

        # Run llm_first.py
        first_args = {
            'dataset_name': args.dataset_name,
            'batch_size': args.batch_size,
            'shuffle': args.shuffle,
            'num_workers': args.num_workers,
            'model_name': args.model_name,
            'gpu': args.gpu,
            'max_new_tokens': args.max_new_tokens,
            'prompt': args.prompt,
            'folder_path':folder_path
        }
        print("Running white_llm_first.py...")
        run_script("white_llm_first.py", first_args)
    else:
        folder_path = f"./results/{args.dataset_name}/results_{args.prompt[0]}/{args.log_file_path}/"
        # log_file_path = args.log_file_path


    second_args = {
        'dataset_name': args.dataset_name,
        'batch_size': 1 if args.interv_type != 'rephrase' else 5,
        'shuffle': args.shuffle,
        'num_workers': 2,
        'model_name': args.model_name,
        'num_mask': args.num_mask,
        'per_num_mask': args.per_num_mask,
        'gpu': args.gpu,
        'max_new_tokens': 20,
        'interv_type': args.interv_type,
        'prompt': args.prompt,
        'log_file_path': args.log_file_path
    }
    # Run the appropriate second script based on interv_type
    if args.interv_type == 'rephrase':
        print("Running white_llm_second_intervention_rephrase.py...")
        run_script("white_llm_second_intervention_rephrase.py", second_args)
    else:
        print("Running white_llm_second_intervention_norephrase.py...")
        run_script("white_llm_second_intervention_norephrase.py", second_args)
    


if __name__ == "__main__":
    main()
