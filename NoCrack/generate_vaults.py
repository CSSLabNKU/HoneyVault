import os

curr_dir = os.path.dirname(os.path.abspath(__file__))

import argparse
from tqdm import tqdm
import lib.pcfg.honeyvault_config as hny_config
from lib.pcfg.helper import convert2group
from lib.pcfg.pcfg import TrainedGrammar
from nocrack_nle import NoCrack_NLE
from toolkits import load_json, save_pickle


def main(nocrack_nle, vaults, args):
    sweetvaults = {}
    sweetcodes = {}
    for id, vault in tqdm(vaults.items(), total=len(vaults)):
        vault_codes = nocrack_nle.encode_vault(vault.copy())
        sweetvaults[id] = [vault]
        sweetcodes[id] = [vault_codes]

        for _ in range(args.num - 1):
            random_sg_does = convert2group(0, 1, hny_config.HONEY_VAULT_GRAMMAR_SIZE)
            random_codes = []
            for _ in range(len(vault)):
                random_codes.append(convert2group(0, 1, hny_config.PASSWORD_LENGTH))
            random_codes.insert(0, random_sg_does)

            sweetvaults[id].append(nocrack_nle.decode_vault(random_codes.copy()))
            sweetcodes[id].append(random_codes)

    result_dir = os.path.join(curr_dir, "results")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    save_pickle(sweetvaults, f"{result_dir}/sweetvaults_{args.num}.pkl")
    save_pickle(sweetcodes, f"{result_dir}/sweetcodes_{args.num}.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num",
        type=int,
        default=1000,
        required=False,
    )
    args = parser.parse_args()

    nocrack_nle = NoCrack_NLE(TrainedGrammar(), "IS-DTE")
    vaults = load_json(f"{curr_dir}/data/pastebin.json")

    main(nocrack_nle, vaults, args)

# python generate_vaults.py --num 1000
