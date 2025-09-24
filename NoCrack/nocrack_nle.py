import os
import sys

curr_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(curr_dir)

from math import log
from collections import Counter
from lib.pcfg.pcfg import make_subgrammar
from lib.pcfg.pcfg import TrainedGrammar


class NoCrack_NLE:
    def __init__(self, tg, selection):
        self.spm = tg
        self.selection = selection

    def subgrammar(self, vault):
        return make_subgrammar(vault, self.spm, self.selection)

    def pw_prob(self, sg, pw):
        return sg.prob(pw)

    def vault_prob(self, vault, sg, selection="IS-DTE"):
        vault_counter = Counter(vault)
        log_vault_prob = 0.0
        for pw, freq in vault_counter.items():
            pw_prob = sg.prob(pw, selection)
            if pw_prob == 0:
                raise ValueError(f"Password {pw} has zero probability.")
            log_vault_prob += freq * log(pw_prob)
        return log_vault_prob

    def encode_vault(self, vault, selection="IS-DTE"):
        sg = make_subgrammar(vault, self.spm, self.selection)
        sg_code = self.spm.encode_grammar(sg)
        vault_code = [sg.encode_pw(pw, selection) for pw in vault]
        vault_code.insert(0, sg_code)
        return vault_code

    def decode_vault(self, vault_code):
        sg_code = vault_code[0]
        sg = self.spm.decode_grammar(sg_code, TG=self.spm)
        vault = [sg.decode_pw(c) for c in vault_code[1:]]
        return vault


def init_nocrack_nle(selection="IS-DTE"):
    tg = TrainedGrammar()
    nocrack_nle = NoCrack_NLE(tg, selection)
    return nocrack_nle
