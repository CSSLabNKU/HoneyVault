import os

curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(curr_dir)
from pathlib import Path

# Original config content
config_content = """[basic_config]
model_path=../../targuess2_train/model/model.txt
sister_password_path=xx
PCFG_path=../../train_data/PCFG_data.txt
markov_path=../../train_data/markov_data.txt
markov_reverse_path=../../train_data/markov_reverse_data.txt
top_psw_path=../../train_data/000webhost_top10000_sorted.txt
log_path=../log/log.txt
output_to_file=xx
maximum_guess_num=5
minimum_prob=1e-18
password_max_len=32
password_min_len=4
maximum_edit_distance=32
minimum_edit_distance=0
print_info_interval=500
verbose=true
predict_password_num=false
predict_timeout=10
check_duplicate=false
output_to_console=false"""

scheme = "vaultguard"
sister_password_folder = Path(f"{root_dir}/{scheme}")


def create_configs():
    # Find all sister_password files
    if not sister_password_folder.exists():
        print(f"Error: Folder {sister_password_folder} does not exist")
        return
    sister_files = list(sister_password_folder.glob(f"sweetvaults_*.txt"))

    for sister_file in sister_files:
        # Create updated config content
        updated_config = config_content.replace(
            "sister_password_path=xx",
            f"sister_password_path={root_dir}/{scheme}/{sister_file.name}",
        ).replace(
            "output_to_file=xx",
            f"output_to_file={curr_dir}/{scheme}_results/{sister_file.name}",
        )

        # Write to a new config file
        output_dir = Path(f"{curr_dir}/{scheme}_config")
        output_dir.mkdir(parents=True, exist_ok=True)
        result_dir = Path(f"{curr_dir}/{scheme}_results")
        result_dir.mkdir(parents=True, exist_ok=True)
        output_file = Path(f"{output_dir}/{sister_file.name}_config.ini")
        with open(output_file, "w") as f:
            f.write(updated_config)

        print(f"Created {output_file}")


if __name__ == "__main__":
    create_configs()
