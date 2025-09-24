import os
import random
import copy

import argparse
from tqdm import tqdm
from toolkits import load_pickle, save_pickle, load_json
from vaultguard_nle import init_vaultguard_nle, HEAdap

curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(curr_dir)


def parse_command():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num",
        required=False,
        type=int,
        default=1000,
    )
    return parser.parse_args()


def main(nle, vaults, args):

    sweetreuses = {}
    sweetvaults = {}
    sweetcodes = {}
    HEAdap(vaults.copy(), args)
    for id, vault in tqdm(vaults.items(), total=len(vaults)):
        copy_vault = copy.deepcopy(vault)
        all_boosted_ngrams = load_pickle(
            f"{curr_dir}/model/{args.num}/boosted_ngrams_{id}.pkl"
        )
        boosted_ngrams = {}
        for prev, next in all_boosted_ngrams[0]:
            if prev not in boosted_ngrams:
                boosted_ngrams[prev] = []
            boosted_ngrams[prev].append(next)
        reuse, code = nle.adaptive_encode_vault(copy_vault, boosted_ngrams)

        sweetvaults[id] = [vault]
        sweetcodes[id] = [code]
        sweetreuses[id] = [reuse]

        for i in range(1, args.num):
            boosted_ngrams = {}
            for prev, next in all_boosted_ngrams[i]:
                if prev not in boosted_ngrams:
                    boosted_ngrams[prev] = []
                boosted_ngrams[prev].append(next)

            random_codes = [
                nle.encoder.dte.padding(nle.encoder.max_code_len)
                for _ in range(len(code))
            ]
            copy_random_codes = copy.deepcopy(random_codes)
            sweetcodes[id].append(random_codes)
            decode_vault = nle.adaptive_decode_vault(
                reuse, copy_random_codes, boosted_ngrams
            )
            sweetvaults[id].append(decode_vault)
            sweetreuses[id].append(reuse)

    result_dir = os.path.join(curr_dir, "results")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    save_pickle(sweetvaults, f"{result_dir}/sweetvaults_{args.num}.pkl")
    save_pickle(sweetcodes, f"{result_dir}/sweetcodes_{args.num}.pkl")
    save_pickle(sweetreuses, f"{result_dir}/sweetreuses_{args.num}.pkl")


if __name__ == "__main__":
    args = parse_command()
    nle = init_vaultguard_nle()
    vaults = load_json(f"{curr_dir}/data/pastebin.json")

    main(nle, vaults, args)

# python generate_vaults.py --num 1000
