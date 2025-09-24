#!/usr/bin/python

"""
This script implements HoneyEncription class for password vauld.
it needs a PCFG in the following format.
"""
from lib2to3.pgen2 import grammar
import os, sys, struct #resource
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from pathlib import Path
p = Path(__name__).resolve() # 取当前模块的绝对路径
parent, root = p.parent, p.parents[1] 
BASE_DIR = os.getcwd() # 返回当前工作目录的绝对路径。
sys.path.append(str(root)) 

from pcfg import pcfg
import honeyvault_config as hny_config
from helper import convert2group

MAX_INT = hny_config.MAX_INT


class DTE(object):
    def __init__(self, grammar=None):
        self.G = grammar
        if not self.G:
            raise Exception("NoSubgrammar")

    def encode(self, lhs, rhs):
        """
        Encode one rule. lhs --> rhs
        """
        assert lhs in self.G, "lhs={} not in self.G".format(lhs)
        a = self.G.encode_rule(lhs, rhs)
        assert a, "NonT ERROR lhs={}, rhs={}, a={}".format(lhs, rhs, a)
        return a

    def decode(self, lhs, pt):
        """
        Decode one rule given a random number pt, and lhs.
        """
        return self.G.decode_rule(lhs, pt)

    def encode_pw(self, pw):
        """
        Encode a password under a the grammar associated with the DTE.
        """
        return self.G.encode_pw(pw)

    def decode_pw(self, P):
        """Given a list of random numbers decode a password under the
        associated grammar with this DTE.
        """
        return self.G.decode_pw(P)

    def __eq__(self, o_dte):
        return self.G == o_dte.G

    def __bool__(self):
        return self.G.is_grammar()


def main():
    grammar = pcfg.TrainedGrammar('D:/honeyvault/nle/nocrack/static/grammar.cfg.gz')
    subgrammar = pcfg.SubGrammar(grammar,) 
    dte = DTE(subgrammar)
    for p in ['(NH4)2Cr2O7', 'iloveyou69']:
        c = dte.encode_pw(p)
        m = dte.decode_pw(c)
        print("After Decoding(encoding of {}): {}".format(p, m))
    for s in range(10000):
        E = []
        E.extend([convert2group(0, 1) for x in range(hny_config.PASSWORD_LENGTH)])
        c_struct = struct.pack('%sI' % len(E), *E)
        m = dte.decode_pw(c_struct)
        print(s, ":", m)


if __name__ == "__main__":
    main()
