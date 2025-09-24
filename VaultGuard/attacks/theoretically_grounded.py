"""
Theoretically-grounded attack against VaultGuard NLE.
"""

import os
import sys

curr_dir = os.path.dirname(os.path.abspath(__file__))
nle_dir = os.path.dirname(curr_dir)
sys.path.append(nle_dir)

import random
import argparse
from tqdm import tqdm
from vaultguard_nle import init_vaultguard_nle
from toolkits import load_pickle, save_json
import os


def load_targuess(file_dir):
    txt_files = [f for f in os.listdir(file_dir) if f.endswith(".txt")]

    if not txt_files:
        raise ValueError(f"No .txt files found in {file_dir}")
    targuess2 = {}
    for txt_file in txt_files:
        file_path = os.path.join(file_dir, txt_file)
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
            if len(parts) == 3:
                bw, pw, prob = parts
                if bw not in targuess2:
                    targuess2[bw] = {}
                targuess2[bw][pw] = float(prob)
    return targuess2


def targuess2_based(pr_v_lst, pr_d_lst):
    """SP24-Duan et al."""
    score = 1.0
    for pr_v, pr_d in zip(pr_v_lst, pr_d_lst):
        score *= (pr_v**2) / pr_d
    return score


def parse_command():
    parser = argparse.ArgumentParser(description="Theoretically grounded attack")
    parser.add_argument(
        "--num",
        required=False,
        type=int,
        default=1000,
        help="Number of sweetvaults.",
    )
    parser.add_argument(
        "--pseudocount",
        type=float,
        required=False,
        default=1,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=2,
        required=False,
    )
    return parser.parse_args()


def main(args):
    nle = init_vaultguard_nle()
    sweet_vaults = load_pickle(f"{nle_dir}/results/sweetvaults_{args.num}.pkl")
    sweet_reuses = load_pickle(f"{nle_dir}/results/sweetreuses_{args.num}.pkl")
    targuess2 = load_targuess(f"{nle_dir}/targuess/targuess2_guess/vaultguard_results")
    D = load_pickle(f"{nle_dir}/data/wishbone_counter.pkl")
    D_size = sum(D.values())

    ranks = {}
    for id, vaults in tqdm(sweet_vaults.items(), total=len(sweet_vaults)):
        score_list = []
        reuses = sweet_reuses[id]
        all_boosted_ngrams = load_pickle(
            f"{nle_dir}/model/{args.num}/boosted_ngrams_{id}.pkl"
        )

        for i in range(args.num):
            boosted_ngrams = {}
            for prev, next_item in all_boosted_ngrams[i]:
                if prev not in boosted_ngrams:
                    boosted_ngrams[prev] = []
                boosted_ngrams[prev].append(next_item)

            vault = vaults[i]
            reuse = reuses[i]
            pr_v_lst = []
            pr_d_lst = []

            for j, pw in enumerate(vault):
                base_idx = nle.encoder.reuse_check(j, reuse)
                if base_idx == None:
                    bw = None
                    pr_v = D.get(pw, 1) / (D_size + 1)
                else:
                    bw = vault[base_idx]
                    if bw not in targuess2.keys():
                        continue
                    if pw not in targuess2[bw].keys():
                        continue
                pr_d = nle.adaptive_prob(bw, pw, boosted_ngrams, args.alpha)
                pr_v_lst.append(pr_v)
                pr_d_lst.append(pr_d)

            score = targuess2_based(pr_v_lst, pr_d_lst)

            if i == 0:
                score_list.append((True, score))
            else:
                score_list.append((False, score))

        random.shuffle(score_list)
        score_list = sorted(score_list, key=lambda x: x[1], reverse=True)
        index = next((k for k, t in enumerate(score_list) if t[0] == True), None)
        ranks[id] = index

    result_dir = os.path.join(nle_dir, f"results")
    os.makedirs(result_dir, exist_ok=True)
    output_path = f"{result_dir}/theoretically_grounded_ranks_{args.num}.json"
    save_json(ranks, output_path)


if __name__ == "__main__":
    args = parse_command()
    main(args)


# python theoretically_grounded.py --num 100
