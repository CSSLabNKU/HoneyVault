"""
Adaptive extra attack against VaultGuard NLE.
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
from toolkits import load_pickle, save_json


def adaptive_extra(vault, boost):
    """USENIX'21-Cheng et al.
    vault: The password vault being tested
    boost: The count of boosted n-grams in the unique passwords of the vault
    note: If there exists a pw in the vault whose n-grams don't appear in the
    boosted n-grams, then this vault can be directly excluded --> score = 0
    """
    p = 0.2
    score = 1.0
    vault_counter = Counter(vault)
    order = 3
    boost_dict = {}
    for pw in vault_counter.keys():
        boost_dict[pw] = []
        pw = " " * order + pw
        len_pw = len(pw)
        for i in range(order, len_pw):
            prefix = pw[i - order : i]
            next = pw[i]
            if prefix not in boost:
                continue
            if next not in boost[prefix]:
                continue
            counts = Counter(boost[prefix])
            if counts[next] >= 2:
                return float("inf")
            boost_dict[pw[order:]].append((prefix, next))
    for pw, count in vault_counter.items():
        m = min([count, len(pw)])
        if len(boost_dict[pw]) < m:
            return 0
        score *= 1 / p**m
        for t in range(len(boost_dict[pw]) + 1, len(pw) + 1):
            score *= (t - m) / t
        score *= count
    return score


def parse_command():
    parser = argparse.ArgumentParser(description="Adaptive extra attack")
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
    sweet_vaults = load_pickle(f"{nle_dir}/results/sweetvaults_{args.num}.pkl")
    ranks = {}
    for id in tqdm(sweet_vaults.keys(), total=len(sweet_vaults)):
        vaults = sweet_vaults[id]
        all_boosted_ngrams = load_pickle(
            f"{nle_dir}/model/{args.num}/boosted_ngrams_{id}.pkl"
        )
        score_list = []

        for i in range(args.num):
            vault = vaults[i]

            boosted_ngrams = {}
            for prev, next_item in all_boosted_ngrams[i]:
                if prev not in boosted_ngrams:
                    boosted_ngrams[prev] = []
                boosted_ngrams[prev].append(next_item)

            score = adaptive_extra(vault, boosted_ngrams)
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
    output_path = f"{result_dir}/adaptive_extra_ranks_{args.num}.json"
    save_json(ranks, output_path)


if __name__ == "__main__":
    args = parse_command()
    main(args)


# python adaptive_extra.py --num 100
