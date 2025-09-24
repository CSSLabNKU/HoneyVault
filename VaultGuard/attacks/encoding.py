"""
Encoding attack against VaultGuard NLE.
"""

import os
import sys
import logging

curr_dir = os.path.dirname(os.path.abspath(__file__))
nle_dir = os.path.dirname(curr_dir)
sys.path.append(nle_dir)

import random
import argparse
from collections import Counter
from tqdm import tqdm
from vaultguard_nle import init_vaultguard_nle
from toolkits import load_pickle, save_json


def encoding(nle, vault, reuse, codes, boosted_ngrams):
    """USENIX21: Weak and strong encoding attack

    Score function
    P_weak: Determine if the generation rules obtained from decoding the code are consistent with the generation rules obtained from encoding the vault. If not consistent, return 0, otherwise return 1.0
    P_strong: Calculate the vault probability, P_strong = P_weak / P_vault, P_vault = ∏_pw P_nle(pw), pw ∈ vault
    """
    w_score = 1
    s_score = 1 / nle.adaptive_vault_prob(reuse, vault, boosted_ngrams)

    for j, (pw, code) in enumerate(zip(vault, codes)):
        base_idx = nle.encoder.reuse_check(j, reuse)
        if base_idx == None:
            encode_rules = nle.encoder.sppm_parse(pw)
            decode_rules = nle.encoder.adaptive_sppm_decode_path(code, boosted_ngrams)
            if encode_rules != decode_rules:
                return 0, 0
        else:
            bw = vault[base_idx]
            encode_rules = nle.encoder.prpm_parse(bw, pw)
            decode_rules = nle.encoder.prpm_decode(bw, code)
            if encode_rules != decode_rules:
                return 0, 0

    return w_score, s_score


def parse_command():
    parser = argparse.ArgumentParser(description="Encoding attack")
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
        default=2,
        required=False,
    )
    return parser.parse_args()


def main(args):
    nle = init_vaultguard_nle()
    sweet_vaults = load_pickle(f"{nle_dir}/results/sweetvaults_{args.num}.pkl")
    sweet_reuses = load_pickle(f"{nle_dir}/results/sweetreuses_{args.num}.pkl")
    sweet_codes = load_pickle(f"{nle_dir}/results/sweetcodes_{args.num}.pkl")

    w_ranks = {}
    s_ranks = {}
    for id in tqdm(sweet_vaults.keys(), total=len(sweet_vaults)):
        all_boosted_ngrams = load_pickle(
            f"{nle_dir}/model/{args.num}/boosted_ngrams_{id}.pkl"
        )
        vaults = sweet_vaults[id]
        reuses = sweet_reuses[id]
        codes = sweet_codes[id]

        w_score_list = []
        s_score_list = []
        for i in range(args.num):
            boosted_ngrams = {}
            for prev, next_item in all_boosted_ngrams[i]:
                if prev not in boosted_ngrams:
                    boosted_ngrams[prev] = []
                boosted_ngrams[prev].append(next_item)

            vault = vaults[i]
            reuse = reuses[i]
            code = codes[i]

            w_score, s_score = encoding(nle, vault, reuse, code, boosted_ngrams)

            if i == 0:
                w_score_list.append((True, w_score))
                s_score_list.append((True, s_score))
            else:
                w_score_list.append((False, w_score))
                s_score_list.append((False, s_score))

        random.shuffle(w_score_list)
        random.shuffle(s_score_list)
        w_score_list = sorted(w_score_list, key=lambda x: x[1], reverse=True)
        s_score_list = sorted(s_score_list, key=lambda x: x[1], reverse=True)
        w_index = next((k for k, t in enumerate(w_score_list) if t[0] == True), None)
        s_index = next((k for k, t in enumerate(s_score_list) if t[0] == True), None)
        w_ranks[id] = w_index
        s_ranks[id] = s_index

    result_dir = os.path.join(nle_dir, f"results")
    os.makedirs(result_dir, exist_ok=True)
    output_path = f"{result_dir}/weak_encoding_ranks_{args.num}.json"
    save_json(w_ranks, output_path)
    output_path = f"{result_dir}/strong_encoding_ranks_{args.num}.json"
    save_json(s_ranks, output_path)


if __name__ == "__main__":

    args = parse_command()
    main(args)

# python encoding.py --num 100
