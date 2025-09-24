"""
Theoretically-grounded attack against NoCrack NLE.
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
from toolkits import save_json, load_pickle


def list_based(pr_v_lst, pr_d_lst):
    """SP24 Duan et al., equation 18
    List model --> 1/3List+1/3PCFG+1/3Markov
    """
    score = 1.0
    for pr_v, pr_d in zip(pr_v_lst, pr_d_lst):
        score *= (pr_v * pr_v) / pr_d
    return score


def parse_command():
    parser = argparse.ArgumentParser(description="Theoretically Grounded attack")
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
            pr_v_lst = []
            pr_d_lst = []

            for pw, freq in vault_counter.items():
                pr_v = D.get(pw, 1) / (D_size + 1)
                pr_d = nle.spm.prob(pw, "IS-DTE")
                pr_v_lst.extend([pr_v] * freq)
                pr_d_lst.extend([pr_d] * freq)

            score = list_based(pr_v_lst, pr_d_lst)

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
    output_path = f"{result_dir}/theoretically_grounded_ranks_{args.num}.json"
    save_json(ranks, output_path)


if __name__ == "__main__":
    args = parse_command()
    main(args)


# python theoretically_grounded.py --num 100 --dataset wishbone
