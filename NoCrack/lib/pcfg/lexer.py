#!/usr/bin/python
"""
This is a general grammar file that assumes the following grammar:
W -> <english-word>L | <name>L
D -> <date> | <phone-no> | [0-9]+
Y -> [^\W]+  # symbol
K -> <keyboard-sequence>
R -> repeat
S -> sequence # 123456, abcdef, ABCDEFG
L -> Capitalize | ALL-UPPER | all-lower | l33t
G -> <some-combination-of-those-NonTs>
"""
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, message="invalid escape sequence")

import os
import string
import sys

BASE_DIR = os.getcwd()
sys.path.append(BASE_DIR)
from dawg import IntDAWG, DAWG
import gzip, re
import random

# from .lexer_helper import Date, RuleSet, ParseTree
from lexer_helper import Date, RuleSet, ParseTree
from helper import open_, load_dawg, isascii  # check_resource,
from honeyvault_config import MIN_COUNT, L33T
from honeyvault_config import MEMLIMMIT, TRAINED_GRAMMAR_FILE

# import resource  # For checking memory usage

# Dictionaries:
# English_30000.dawg
# facebook-firstnames-withcount.dawg
# facebook-lastnames-withcount.dawg


"""
这种结构特别适用于需要快速查找和映射复杂关系的场景。例如，在一个词法分析器（lexer）中，可能需要根据标识符的类型和长度来确定其属于哪个类别或执行哪种操作。NonT_length2classmap提供了一种高效的方式来存储和访问这种类型的映射关系。
"""
NonT_length2classmap = {
    "W": {
        "1": [1, 2],
        "2": [3, 3],
        "3": [4, 4],
        "4": [5, 5],
        "5": [6, 6],
        "6": [7, 7],
        "7": [8, 8],
        "8": [9, 9],
        "9": [9, 30],
    },
    "D": {"1": [1, 1], "2": [2, 3], "3": [4, 6], "4": [7, 9], "5": [10, 30]},
    "Y": {"1": [1, 1], "2": [2, 30]},
}


def get_nont_class(nt, word):
    A = NonT_length2classmap.get(nt, {})
    n = len(word)
    for k, v in list(A.items()):
        if n >= v[0] and n <= v[1]:
            return k


class NonT(object):  # baseclass
    def __init__(self):
        # self.sym = 'G'
        self.prob = 0.0
        self.prod = ""

    def symbol(self):
        return self.sym

    def probability(self):
        return self.prob

    def production(self):
        return self.prod

    def __str__(self):
        p_str = (
            [str(p) for p in self.prod] if isinstance(self.prod, list) else self.prod
        )

        return "%s: %s (%g)" % (self.sym, p_str, self.prob)

    def parse_tree(self):
        p_tree = ParseTree()
        if isinstance(self.prod, str):
            return self.prod
        elif isinstance(self.prod, list):
            for p in self.prod:
                p_tree.add_rule((p.sym, p.parse_tree()))
        else:
            return self.prod.parse_tree()
        return p_tree

    def rule_set(self):
        rs = RuleSet()
        if isinstance(self, NonT):
            rs.add_rule("G", self.sym)
        if isinstance(self.prod, str):
            rs.add_rule(self.sym, self.prod)
        elif isinstance(self.prod, list):
            for p in self.prod:
                rs.update_set(p.rule_set())
        else:
            return self.prod.rule_set()
        return rs

    def __bool__(self):
        return bool(self.prod) and bool(self.prob)


class NonT_L(NonT):
    sym, prod, prob = "L", "", 0.0

    def __init__(self, v, w):
        """
        v = w.lower()
        """
        super(NonT_L, self).__init__()
        self.prod = (
            "UPPER"
            if v.upper() == w
            else "lower" if v.lower() == w else "Caps" if v.title() == w else "l33t"
        )
        self.r = w
        self.l = v
        if self.prod == "l33t":
            # v 中由于 l33t 转换后的字符数量
            c = len([1 for cl, cr in zip(self.l, self.r) if cl != cr.lower()])
            # 特定的计算进行 l33t 转换的概率
            self.prob = 1 - c / len(self.r)
        else:
            self.prob = 1.0

    def parse_tree(self):
        p_tree = ParseTree()
        p_tree.add_rule(("L", self.prod))
        L = ["L_%s" % c for c in self.l]
        if self.prod == "l33t":
            p_tree.add_rule(("l33t", list(zip(L, self.r))))
        return p_tree

    def rule_set(self):
        rs = RuleSet()
        rs.add_rule("L", self.prod)
        if self.prod == "l33t":
            for c, d in zip(self.l, self.r):
                rs.add_rule("L_%s" % c, d)
        return rs

    def __str__(self):
        return "NonT_L: ({}, {})".format(self.l, self.r)


class NonT_W(NonT):
    sym, prod, prob = "W", "", 0.0
    thisdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # word_dawg = load_dawg('{}/data/English_30000.dawg.gz'.format(thisdir))
    fname_dawg = load_dawg(
        "{}/data/facebook-firstnames-withcount.dawg.gz".format(thisdir)
    )
    lname_dawg = load_dawg(
        "{}/data/facebook-lastnames-withcount.dawg.gz".format(thisdir)
    )
    # total_f = word_dawg['__total__'] + fname_dawg['__total__'] + lname_dawg['__total__']

    l33t_replaces = DAWG.compile_replaces(L33T)

    def __init__(self, word):
        # super(NonT_W, self).__init__()
        w = str(word.lower())
        dawg = []
        for d in [self.word_dawg, self.fname_dawg, self.lname_dawg]:
            k = d.similar_keys(w, self.l33t_replaces)
            if k:
                dawg.append((d, k[0]))
        if dawg:
            v = list(set([d[1] for d in dawg]))
            if len(v) > 1 or not v[0].isalpha():
                return
            v = v[0]
            f = sum([d[0][v] for d in dawg])
            self.prod = v
            self.sym = "W%s" % get_nont_class("W", v)
            self.L = NonT_L(v, word)
            self.prob = self.L.prob * float(f) / self.total_f

    def parse_tree(self):
        pt = ParseTree()
        pt.add_rule((self.sym, self.prod))
        pt.extend_rules(self.L.parse_tree())
        return pt

    def rule_set(self):
        rs = RuleSet()
        rs.add_rule(self.sym, self.prod)
        rs.update_set(self.L.rule_set())
        return rs

    def __str__(self):
        return "%s: %s<%s> (%g)" % (self.sym, self.prod, self.L, self.prob)


class NonT_D(NonT):
    sym, prod, prob = "D", "", 0.0

    def __init__(self, w):
        # super(NonT_D, self).__init__()
        if w.isdigit():
            self.prod = w
            self.prob = 0.001
            self.sym = "D%s" % get_nont_class("D", w)
        d = Date(w)
        if d:
            self.sym = "T"
            self.prod = d
            self.prob = 10 ** (len(w) - 8)

    def parse_tree(self):
        if isinstance(self.prod, str):
            return ParseTree(self.sym, self.prod)
        else:
            return self.prod.parse_tree()

    def rule_set(self):
        if isinstance(self.prod, str):
            return RuleSet(self.sym, self.prod)
        else:
            return self.prod.rule_set()


class NonT_R(NonT):  # repeat
    sym, prod, prob = "R", "", 0.0

    def __init__(self, w):
        x = len(set(w)) / float(len(w))
        if x < 0.2:
            self.prob = 1 - float(x) / len(w)
            self.prod = w


class NonT_Q(NonT):
    sym, prod, prob = "Q", "", 0.0
    ascii_u = string.ascii_uppercase
    ascii_l = string.ascii_uppercase
    pass


class NonT_K(NonT):
    pass


class NonT_Y(NonT):
    sym, prod, prob = "Y", "", 0.0
    regex = r"^[\W_]+$"

    def __init__(self, word):
        # super(NonT_Y, self).__init__()
        if re.match(self.regex, word):
            self.prod = word
            self.prob = 0.01
            self.sym = "Y%s" % get_nont_class("Y", word)


class NonT_combined(NonT):
    sym, prod, prob = "C", "", 0.0

    def __init__(self, *nont_set):
        for p in nont_set:
            if not p:
                return
        self.sym = ",".join([x.symbol() for x in nont_set])
        self.prod = []
        for p in nont_set:
            if isinstance(p.production(), list):
                self.prod.extend(p.production())
            else:
                self.prod.append(p)
        self.prob = prod([p.probability() for p in nont_set])


def get_all_gen_rules(word):
    if not word:
        return None
    NonT_set = [NonT_W, NonT_D, NonT_Y, NonT_R]
    rules = [x for x in [f(word) for f in NonT_set] if x]
    if rules:
        return max(rules, key=lambda x: x.probability())


def prod(args):
    p = 1.0
    for i in args:
        p = p * i
    return p


def join_rules(*args):
    for a in args:
        if not a:
            return None
    return NonT_combined(*args)


def parse(word, selection):
    A = {}
    for j in range(len(word)):
        for i in range(len(word) - j):
            A[(i, i + j)] = get_all_gen_rules(word[i : j + i + 1])
            t = [A[(i, i + j)]]
            t.extend(
                [NonT_combined(A[(i, k)], A[(k + 1, i + j)]) for k in range(i, i + j)]
            )
            t = [x for x in t if x]
            if t:
                # 选择一个概率最大的最为这个区间的解析结果
                if selection == "IS-DTE":
                    A[(i, i + j)] = max(t, key=lambda x: x.probability())
                # 按概率分布选择一个解析结果
                elif selection == "IS-PMTE":
                    probs = [x.probability() for x in t]
                    A[(i, i + j)] = random.choices(t, weights=probs, k=1)[0]
            else:
                A[(i, i + j)] = NonT()
                print("Not sure why it reached here. But it did!")
                print(i, j, word[i : i + j + 1])
                # exit(0)
    return NonT_combined(A[(0, len(word) - 1)])
