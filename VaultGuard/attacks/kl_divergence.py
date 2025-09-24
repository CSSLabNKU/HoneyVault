"""
KL divergence attack against VaultGuard NLE.
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
from vaultguard_nle import init_vaultguard_nle
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
        required=False,
        type=int,
        default=100,
        help="Number of sweetvaults.",
    )
    parser.add_argument(
        "--pseudocount",
        type=float,
        required=False,
        default=1.000,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        required=False,
    )
    return parser.parse_args()


def main(args):
    nle = init_vaultguard_nle()

    sweet_vaults = load_pickle(f"{nle_dir}/results/sweetvaults_{args.num}.pkl")
    sweet_reuses = load_pickle(f"{nle_dir}/results/sweetreuses_{args.num}.pkl")

    ranks = {}
    for id in tqdm(sweet_vaults.keys(), total=len(sweet_vaults)):
        all_boosted_ngrams = load_pickle(
            f"{nle_dir}/model/{args.num}/boosted_ngrams_{id}.pkl"
        )
        vaults = sweet_vaults[id]
        reuses = sweet_reuses[id]
        score_list = []

        for i in range(args.num):
            boosted_ngrams = {}
            for prev, next_item in all_boosted_ngrams[i]:
                if prev not in boosted_ngrams:
                    boosted_ngrams[prev] = []
                boosted_ngrams[prev].append(next_item)

            vault = vaults[i]
            reuse = reuses[i]
            vault_counter = Counter(vault)
            total = sum(vault_counter.values())
            p = []
            q = []

            for pw, freq in vault_counter.items():
                p.append(freq / total)
                j = vault.index(pw)
                base_idx = nle.encoder.reuse_check(j, reuse)
                if base_idx == None:
                    bw = None
                else:
                    bw = vault[base_idx]
                q.append(nle.adaptive_prob(bw, pw, boosted_ngrams, args.alpha))

            score = kl_divergence(p, q)

            if i == 0:
                score_list.append((True, score))
            else:
                score_list.append((False, score))

        random.shuffle(score_list)
        score_list = sorted(score_list, key=lambda x: float(x[1]), reverse=True)
        index = next((k for k, t in enumerate(score_list) if t[0] == True), None)
        ranks[id] = index

    result_dir = os.path.join(nle_dir, f"results")
    os.makedirs(result_dir, exist_ok=True)
    output_path = f"{result_dir}/kl_divergence_ranks_{args.num}.json"
    save_json(ranks, output_path)


if __name__ == "__main__":
    args = parse_command()
    main(args)

# python kl_divergence.py --num 100
