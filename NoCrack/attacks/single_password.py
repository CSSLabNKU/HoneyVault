"""
Single password attack against NoCrack NLE.
"""

import os
import sys

curr_dir = os.path.dirname(os.path.abspath(__file__))
nle_dir = os.path.dirname(curr_dir)
sys.path.append(nle_dir)

import random
import argparse
from tqdm import tqdm
from collections import Counter
from nocrack_nle import init_nocrack_nle
from toolkits import load_pickle, save_json


def single_password(f_a, Pr_decoy, D_size, a=1, f_d=5):
    """USENIX21: Single-password attack

    Score function
    P_opt = Π_pw P_sp(pw), P_sp(pw) = Pr_smooth(pw) / Pr_decoy(pw)
    if f_a(pw) <= 5 and P_sp(pw) > 1: P_sp(pw) = 1; else: P_sp(pw) remains unchanged

    Details:
    Pr_decoy: can be calculated by the single-password models in honey vault args.schemes
    f_a(pw) : the absolute frequency of pw in the training set D
    D_size  : the size of the training set D
    a = 1   : is a smoothing parameter
    f_d = 5 : is a parameter representing the demarcation line between high-frequency and low-frequency passwords
    """
    score = 1.0
    for fa, fd in zip(f_a, Pr_decoy):
        fr = (fa + a) / (D_size + a)
        p = fr / fd
        if fa <= f_d and p > 1:
            score *= 1
        else:
            score *= p
        score *= p
    return score


def parse_command():
    parser = argparse.ArgumentParser(description="Single password attack")
    parser.add_argument(
        "--num",
        required=False,
        type=int,
        default=100,
        help="Number of sweetvaults.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["wishbone", "rockyou"],
        default="wishbone",
        required=False,
        help="Dataset for attacker.",
    )
    return parser.parse_args()


def main(args):
    nle = init_nocrack_nle()
    sweet_vaults = load_pickle(f"{nle_dir}/results/sweetvaults_{args.num}.pkl")
    D = load_pickle(f"{nle_dir}/data/{args.dataset}_counter.pkl")
    D_size = sum(D.values())

    ranks = {}
    for id in tqdm(sweet_vaults.keys(), total=len(sweet_vaults)):
        vaults = sweet_vaults[id]
        score_list = []

        for i in range(args.num):
            vault = vaults[i]
            vault_counter = Counter(vault)
            Pr_decoy = []
            f_a = []

            for pw, freq in vault_counter.items():
                pr_real = D.get(pw, 0)
                pr_decoy = nle.spm.prob(pw, "IS-DTE")
                f_a.extend([pr_real] * freq)
                Pr_decoy.extend([pr_decoy] * freq)

            score = single_password(f_a, Pr_decoy, D_size)

            if i == 0:
                score_list.append((True, score))
            else:
                score_list.append((False, score))

        random.shuffle(score_list)
        score_list = sorted(score_list, key=lambda x: x[1], reverse=True)
        index = next((k for k, t in enumerate(score_list) if t[0] == True), None)
        ranks[id] = index

    result_dir = os.path.join(nle_dir, "results")
    os.makedirs(result_dir, exist_ok=True)
    output_path = f"{result_dir}/single_password_ranks_{args.num}.json"
    save_json(ranks, output_path)


if __name__ == "__main__":
    args = parse_command()
    main(args)

# python single_password.py --num 100 --dataset wishbone
