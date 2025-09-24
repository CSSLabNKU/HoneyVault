import os

curr_dir = os.path.dirname(os.path.abspath(__file__))
import argparse
from tqdm import tqdm
from toolkits import load_pickle
from vaultguard_nle import init_vaultguard_nle


def save_txt(data, file_path):
    with open(file_path, "w") as f:
        for pw1, pw2 in data:
            f.write(f"\t{pw1}\t{pw2}\n")


def load_txt(file_path):
    with open(file_path, "r") as f:
        data = {}
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            pw1, pw2, prob = parts
            data[pw1][pw2] = float(prob)
        return data


def process_sweetvaults(nle, sweetvaults, sweetreuses):
    sister_passwords = {}
    for id in tqdm(sweetvaults.keys(), total=len(sweetvaults)):
        sister_passwords[id] = {}
        vaults = sweetvaults[id]
        reuses = sweetreuses[id]
        for vault, reuse in zip(vaults, reuses):
            for j, pw in enumerate(vault):
                base_idx = nle.encoder.reuse_check(j, reuse)
                if base_idx != None:
                    bw = vault[base_idx]
                    if bw == pw:
                        continue
                    if bw not in sister_passwords[id]:
                        sister_passwords[id][bw] = []
                    sister_passwords[id][bw].append(pw)

    return sister_passwords


def preprocess(nle, sweetvaults, sweetreuses, args):
    sister_passwords = process_sweetvaults(nle, sweetvaults, sweetreuses)
    all_sister_passwords = []
    for id, bw_pws in sister_passwords.items():
        for bw, pws in bw_pws.items():
            for pw in pws:
                all_sister_passwords.append((bw, pw))
    chunk_size = 40000
    for i in range(0, len(all_sister_passwords), chunk_size):
        chunk = all_sister_passwords[i : i + chunk_size]
        save_txt(
            chunk,
            f"{curr_dir}/targuess/vaultguard/sweetvaults_{i // chunk_size + 1}.txt",
        )


def parse_command():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num",
        required=False,
        type=int,
        default=1000,
    )
    output_dir = f"{curr_dir}/targuess/vaultguard"
    os.makedirs(output_dir, exist_ok=True)
    return parser.parse_args()


def main(args):
    nle = init_vaultguard_nle()
    sweetvaults = load_pickle(f"{curr_dir}/results/sweetvaults_{args.num}.pkl")
    sweetreuses = load_pickle(f"{curr_dir}/results/sweetreuses_{args.num}.pkl")

    preprocess(nle, sweetvaults, sweetreuses, args)


if __name__ == "__main__":
    args = parse_command()
    main(args)

# python convert2targuess.py --num 100
