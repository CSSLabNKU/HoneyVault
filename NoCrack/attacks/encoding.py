"""
Weak & Strong encoding attack against NoCrack NLE.
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
from nocrack_nle import init_nocrack_nle


def check(pt1, pt2):
    if len(pt1) != len(pt2):
        return 0
    else:
        len_pt = len(pt1)
        for i in range(len_pt):
            if pt1[i][0] != pt2[i][0] or pt1[i][1] != pt2[i][1]:
                return 0
        return 1


def encoding(vault, code, nle):
    """USENIX21: W and s encoding attack

    Score function
    P_w: Determines whether the generation rules decoded from the code are consistent with the generation rules encoded from the vault. Returns 0 if inconsistent, otherwise returns 1.0.
    P_s: Calculates the probability of the vault, P_s = P_w / P_vault, where P_vault = ∏_pw P_nle(pw), pw ∈ vault
    """
    w_score = 1
    s_score = 1
    sg_code = code.pop(0)
    sg = nle.spm.decode_grammar(sg_code, TG=nle.spm)
    for i in range(len(vault)):
        pw = vault[i]
        pw_code = code[i]
        encode_rules = sg.l_parse_tree(pw, "IS-DTE")
        decode_rules = sg.decode_path(pw_code)
        encode_rules_counter = Counter(encode_rules)
        decode_rules_counter = Counter(decode_rules)
        for r, freq in encode_rules_counter.items():
            if r not in decode_rules_counter:
                return 0, 0
            if freq != decode_rules_counter[r]:
                return 0, 0
    s_score = w_score / nle.vault_prob(vault, sg)
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
    return parser.parse_args()


def main(args):
    nle = init_nocrack_nle()
    sweet_vaults = load_pickle(f"{nle_dir}/results/sweetvaults_{args.num}.pkl")
    sweet_codes = load_pickle(f"{nle_dir}/results/sweetcodes_{args.num}.pkl")

    w_ranks = {}
    s_ranks = {}
    for id in tqdm(sweet_vaults.keys(), total=len(sweet_vaults)):
        vaults = sweet_vaults[id]
        codes = sweet_codes[id]

        w_score_list = []
        s_score_list = []
        for i in range(args.num):
            vault = vaults[i]
            code = codes[i]

            w_score, s_score = encoding(vault, code, nle)

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

    result_dir = os.path.join(nle_dir, "results")
    os.makedirs(result_dir, exist_ok=True)
    save_json(w_ranks, f"{result_dir}/weak_encoding_ranks_{args.num}.json", 4)
    save_json(s_ranks, f"{result_dir}/strong_encoding_ranks_{args.num}.json", 4)


if __name__ == "__main__":
    args = parse_command()
    main(args)


# python encoding.py --num 100
