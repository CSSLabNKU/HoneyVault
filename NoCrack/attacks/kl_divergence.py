"""
KL divergence attack against NoCrack NLE.
"""

import os
import sys

curr_dir = os.path.dirname(os.path.abspath(__file__))
nle_dir = os.path.dirname(curr_dir)
sys.path.append(nle_dir)

import random
import argparse
from math import log
from tqdm import tqdm
from collections import Counter
from nocrack_nle import init_nocrack_nle
from toolkits import load_pickle, save_json


def kl_divergence(p, q):
    """CCS'16: KL divergence attack
    p: based on relative frequencies, e.g., [0.1, 0.4,..., 0.3]
    q: based on probabilities calculated by NLE, e.g., [0.1, 0.2,..., 0.2]
    Returns:[0,inf], The closer to 0 is, the more similar the two distributions are

    NOTE: Calculte KL divergence on the entire vault instead of unique passwords.
    """
    score = 0.0
    for pi, qi in zip(p, q):
        score += pi * log(pi / qi)
    return score


def parse_command():
    parser = argparse.ArgumentParser(description="KL divergence attack")
    parser.add_argument(
        "--num",
        required=True,
        type=int,
        default=1000,
        help="Number of sweetvaults.",
    )
    return parser.parse_args()


def main(args):
    nle = init_nocrack_nle()
    sweet_vaults = load_pickle(f"{nle_dir}/results/sweetvaults_{args.num}.pkl")

    ranks = {}
    for id in tqdm(sweet_vaults.keys(), total=len(sweet_vaults)):
        vaults = sweet_vaults[id]

        score_list = []
        for i in range(args.num):
            vault = vaults[i]
            sg = nle.subgrammar(vault)
            vault_counter = Counter(vault)
            total = sum(vault_counter.values())
            p = []
            q = []

            for pw, freq in vault_counter.items():
                p.append(freq / total)
                q.append(sg.prob(pw, "IS-DTE"))

            score = kl_divergence(p, q)

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
    output_path = f"{result_dir}/kl_divergence_ranks_{args.num}.json"
    save_json(ranks, output_path)


if __name__ == "__main__":
    args = parse_command()
    main(args)


# python kl_divergence.py --num 100
