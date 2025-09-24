#!/usr/bin/python
"""
This is a pcfg of almost the following grammar:
W -> <english-word>L | <name>L
D -> <date> | <phone-no> | [0-9]+
Y -> [^\W]+  # symbol
K -> <keyboard-sequence>
R -> repeat
S -> sequence # 123456, abcdef, ABCDEFG
L -> Capitalize | ALL-UPPER | all-lower | l33t
G -> DG | YG | WG | RG | SG | KG | D | Y | W | R | S | K
"""

import os
import sys

import warnings

warnings.filterwarnings(
    "ignore", category=SyntaxWarning, message="invalid escape sequence"
)

from sympy import re

curr_file_path = os.path.abspath(__file__)
curr_dir = os.path.dirname(curr_file_path)
if curr_dir not in sys.path:
    sys.path.append(curr_dir)
import string

BASE_DIR = os.getcwd()
from dawg import IntDAWG, DAWG
import json
from lexer_helper import Date, RuleSet, ParseTree
from lexer import NonT_L, get_nont_class
from helper import (
    open_,
    getIndex,
    convert2group,
    print_err,
    bin_search,
    print_once,
    random,
    whatchar,
    DEBUG,
    load_pickle,
    save_pickle,
    joe_clean_vaultdata,
)
import honeyvault_config as hny_config
from collections import OrderedDict


class TrainedGrammar(object):
    l33t_replaces = DAWG.compile_replaces(hny_config.L33T)  # TODO: 作用

    def __init__(
        self, g_file=hny_config.TRAINED_GRAMMAR_FILE, g_itself=None, cal_cdf=False
    ):
        self.cal_cdf = cal_cdf
        if g_file == "xxx":
            self.G = g_itself
            Wlist = [x for k, v in list(self.G.items()) for x in v if k.startswith("W")]

            """
            k = W3; v = {good:12, ok: 15, cat: 20}
            Wlist = [good,ok,cat,dog,'__total__']
            """
            self.date = Date()
            self.Wdawg = IntDAWG(Wlist)

        else:
            self.load(g_file)  # 调用 load 函数
        # TODO: T_Y, T_d, T_m, T_y 这些规则是什么意思，为什么要筛选出来; L_x 由l33t_replaces表示？
        # NonT_set = ['D1', 'D2', 'D3', 'D4', 'D5', 'G', 'L', 'R', 'T', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W9', 'Y1', 'Y2']
        self.NonT_set = [x for x in list(self.G.keys()) if x.find("_") < 0]

    def load(self, filename):
        self.G = json.load(open_(filename), object_pairs_hook=OrderedDict)
        self.G["Y1"]["\\"], self.G["Y1"]['"'] = 225, 2
        for k, v in list(self.G.items()):
            if self.cal_cdf:
                print_err("Calculating CDF!")
                """
                k = W3; v = {good:12, ok: 15, cat: 20}
                v = {good:12, ok:27,cat:47,__total__:47}
                """
                lf = 0
                for l, f in list(v.items()):
                    v[l] += lf
                    lf += f
                v["__total__"] = lf
            else:
                v["__total__"] = sum(v.values())

        # Create dawg/trie of the Wlist items for fast retrieval
        """
        k = W3; v = {good:12, ok: 15, cat: 20}
        Wlist = [good,ok,cat,dog,'__total__']
        """
        Wlist = [x for k, v in list(self.G.items()) for x in v if k.startswith("W")]
        self.date = Date()  # TODO: 作用
        self.Wdawg = IntDAWG(Wlist)  # TODO: 作用

    def get_prob(self, l, r):
        f = self.G.get(l, {}).get(r, 0)
        tot = self.G.get(l, {}).get("__total__", 1e-3)
        return max(float(f) / tot, 0.0)

    def prob(self, pw, selection):
        pt = self.l_parse_tree(pw, selection)
        proba = 1.0
        for it in pt:
            proba *= self.get_prob(it[0], it[1])
        if proba == 0:
            proba = 1e-10
        return proba

    def get_freq(self, l, r):
        return self.G.get(l, {}).get(r, 0)

    def isNonTerm(self, lhs):  # this means given lhs, rhs will be in NonT
        return lhs in self.NonT_set

    def get_actual_NonTlist(self, lhs, rhs):
        if lhs == "G":
            # Don't include, "W1,G", "D1,G", "Y1,G" etc.
            if rhs.endswith(",G"):
                return []
            # "D1,D1,W1,D2" -> [D1,D1,W1,D2]
            return rhs.split(",")
        elif lhs == "T":
            # "Y,m,d" -> [T_Y, T_m, T_d]
            return ["%s_%s" % (lhs, c) for c in (rhs.split(",") if "," in rhs else rhs)]
        elif lhs == "L":
            return ["%s_%s" % (lhs, c) for c in rhs]
        elif lhs in ["W", "D", "Y", "R", "K"]:
            return []
        else:
            return []

    def get_W_rule(self, word):
        w = str(word.lower())
        # 查找与 w 在 l33t_replaces 替换下相似的单词, w='a@a', k=['a@a', 'aaa']
        k = self.Wdawg.similar_keys(w, self.l33t_replaces)
        if k:
            k = k[0]  # 选择 w
            L = NonT_L(k, word)
            # 找到这个word具体归属在哪一类？比如返回的是 W4
            sym = "W%s" % get_nont_class("W", k)
            try:
                p = self.get_prob(sym, k)
                # 计算一下概率
            except KeyError as ex:
                print(k, sym, ex)
                raise KeyError(ex)
            # ('W4', [('money', <pcfg.lexer.NonT_L object at 0x000002D954C43C10>)], 0.005865458813527201)
            return sym, [(k, L)], p

    def get_T_rule(self, word):
        T = self.date.IsDate(word)
        if T:
            p = 10 ** (len(word))
            for r in T.tree:
                p *= self.get_prob(*r)
            p *= self.get_prob(*(T.get_rule()))
            # ('T', [('19990714', [('T_Y', '1999'), ('T_m', '07'), ('T_d', '14')])], 0.8667281485277206)
            return "T", [(word, T)], p

    def get_all_matches(self, word, selection="IS-DTE"):
        import random

        rules = []
        for nt in self.NonT_set:
            # NonT_set = ['D1', 'D2', 'D3', 'D4', 'D5', 'G', 'L', 'R', 'T', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W9', 'Y1', 'Y2']
            if nt.startswith("G"):  # 'G' should not be considered here
                continue
            if nt.startswith("W"):
                l = self.get_W_rule(word)
                if l:
                    rules.append(l)
            elif nt == "T":
                l = self.get_T_rule(word)
                if l:
                    rules.append(l)
            else:
                f = self.G[nt].get(word, 0)
                if f > 0:
                    rules.append((nt, [word], float(f) / self.G[nt]["__total__"]))
        # 过滤掉rules列表中的空元素和概率为0的规则
        rules = [x for x in rules if x and x[-1]]
        if rules:
            # 返回概率最高的规则
            # return max(rules, key=lambda x: x[-1])
            # 按概率分布返回规则
            # probs = [x[-1] for x in rules]
            # return random.choices(rules, weights=probs, k=1)[0]
            # 选择一个概率最大的最为这个区间的解析结果
            if selection == "IS-DTE":
                return max(rules, key=lambda x: x[-1])
            # 按概率分布选择一个解析结果
            elif selection == "IS-PMTE":
                probs = [x[-1] for x in rules]
                return random.choices(rules, weights=probs, k=1)[0]
            else:
                raise ValueError("Unknown selection method!")

    def join(self, r, s):
        """
        [W4->pass, W4->word] => [W4,W4->password]
        """

        def not_startswith_L_T(x):
            return x and not (x[0].startswith("L_") or x[0].startswith("T_"))

        if not_startswith_L_T(s) and not_startswith_L_T(r):
            k = ",".join([r[0], s[0]])
            p = r[-1] * s[-1]
            a = r[1] + s[1]
            return (k, a, p)

    def random_parse(self, word, try_num=3):
        """
        Returns a random parse of the word following the grammar.
        不用管这个函数
        """
        # First- rejection sampling, most inefficient version
        # break the word into random parts and then see if that parse exist
        print("\n^^^^^^^^^^^_______________^^^^^^^^^^^^^^")
        if try_num < 0:
            print("I am very sorry. I could not parse this :(!!")
            return None
        # NO IDEA HOW TO randomly pick a parse tree!! @@TODO
        raise ValueError("Not implemented")

    def parse(self, word, selection):
        """
        根据概率值，给出word字符串的最优解析结果。
        """
        import random

        A = {}
        if not word:
            return ()
        # parse先将word解析成小的子字符串，在各个区间将小的子字符串组合成大的字符串
        # i 控制子字符串的起始位置, j 控制子字符串的长度
        for j in range(len(word)):
            for i in range(len(word) - j):
                A[(i, i + j)] = self.get_all_matches(word[i : j + i + 1])
                t = [A[(i, i + j)]]
                t.extend(
                    [self.join(A[(i, k)], A[(k + 1, i + j)]) for k in range(i, i + j)]
                )
                if t:
                    # 选择一个概率最大的最为这个区间的解析结果
                    if selection == "IS-DTE":
                        A[(i, i + j)] = max(t, key=lambda x: x[-1] if x else 0)
                    # 按概率分布选择一个解析结果
                    elif selection == "IS-PMTE":
                        probs = [x[-1] if x else 0 for x in t]
                        A[(i, i + j)] = random.choices(t, weights=probs, k=1)[0]
                    else:
                        raise ValueError("Unknown selection method!")
                else:
                    A[(i, i + j)] = ()
        return A[(0, len(word) - 1)]

    @staticmethod
    def default_parse_tree(word):
        """
        Returns the default parse of a word. Default parse is
        G -> W1,G | D1,G | Y1,G | W1 | D1 | Y1
        This parses any string over the allowed alphabet
        returns a l-o-r traversed parse tree
        # TODO: 这个函数是否对应论文中的 catch-all rule ?
        """
        pt = ParseTree()
        n = len(word)
        for i, c in enumerate(word):
            r = whatchar(c) + "1"
            if i < n - 1:
                r = r + ",G"
            pt.add_rule(("G", r))
            pt.add_rule((r[:2], c.lower()))
            if r.startswith("W"):
                nont_l = NonT_L(c, c)
                pt.extend_rules(nont_l.parse_tree())

        return pt

    def l_parse_tree(self, word, selection):
        """leftmost parse-tree
        返回一个word的生成规则树，word='admiin123',
        pt={
                "D2":{
                    "123":1
                },
                "G":{
                    "W4,D2":1
                },
                "L":{
                    "lower":1
                },
                "W4":{
                    "admin":1
                }
            }
        """
        pt = ParseTree()
        p = self.parse(word, selection)
        if not p:
            print("Failing at {!r}".format(word))
            return pt

        # TODO: 无法通过G来解析word?进入catch-all rule环节？
        if p[0] not in self.G["G"]:
            return self.default_parse_tree(word)

        pt.add_rule(("G", p[0]))
        for l, each_r in zip(p[0].split(","), p[1]):
            if isinstance(each_r, str):
                pt.add_rule((l, each_r))
            elif l.startswith("W"):
                pt.add_rule((l, each_r[0]))
                L_parse_tree = each_r[1].parse_tree()
                pt.add_rule(L_parse_tree[0])
                if len(L_parse_tree.tree) > 1:
                    pt.tree.extend(L_parse_tree[1][1])
            elif l == "T":
                p = each_r[1]
                rule_name = ",".join([r[0].replace("T_", "") for r in p])
                pt.add_rule((l, rule_name))
                pt.extend_rules(p)
            else:
                print("Something is severely wrong")
        return pt

    def pw2rules(self, word):
        return self.l_parse_tree(word)

    def check(self, pt1, pt2):
        if len(pt1) != len(pt2):
            return 0
        else:
            len_pt = len(pt1)
            for i in range(len_pt):
                if pt1[i][0] != pt2[i][0] or pt1[i][1] != pt2[i][1]:
                    return 0
            return 1

    def rule_set(self, word):
        """
        # TODO: 返回值？
        """
        rs = RuleSet()
        pt = self.l_parse_tree(word)
        for p in pt.tree:
            rs.add_rule(*p)
        return rs

    def encode_rule(self, l, r):
        rhs_dict = self.G[l]
        try:
            i = list(rhs_dict.keys()).index(r)
            if DEBUG:
                c = list(rhs_dict.keys())[i]
                assert c == r, "The index is wrong"
        except ValueError:
            raise ValueError("'{}' not in the rhs_dict (l: '{}')".format(r, l))
        l_pt = sum(list(rhs_dict.values())[:i])
        r_pt = l_pt + rhs_dict[r] - 1
        assert (
            l_pt <= r_pt
        ), "Rule with zero freq! rhs_dict[{}] =  {} (l={})\n{}".format(
            r, rhs_dict, l, self.G
        )
        return convert2group(random.randint(l_pt, r_pt), rhs_dict["__total__"])

    def decode_rule(self, l, p):
        rhs_dict = self.G[l]
        if not rhs_dict:
            return ""
        assert "__total__" in rhs_dict, "__total__ not in {!r}, l={!r}".format(
            rhs_dict, l
        )
        p %= rhs_dict["__total__"]

        if self.cal_cdf:
            if len(rhs_dict) > 1000:
                print_once(l, len(rhs_dict))
            return bin_search(list(rhs_dict.items()), p, 0, len(rhs_dict))
        for k, v in list(rhs_dict.items()):
            if p < v:
                return k
            else:
                p -= v
        print("Allas could not find.", l, p)

    def decode_l33t(self, w, iterp):
        l = self.decode_rule("L", next(iterp))
        if l == "Caps":
            return w.capitalize()
        elif l == "lower":
            return w.lower()
        elif l == "UPPER":
            return w.upper()
        else:
            nw = "".join([self.decode_rule("L_%s" % c, next(iterp)) for c in w])
            return nw

    def encode_pw(self, pw, selection):
        pt = self.l_parse_tree(pw, selection)
        code_g = [self.encode_rule(*p) for p in pt]
        extra = hny_config.PASSWORD_LENGTH - len(code_g)
        code_g.extend(convert2group(0, 1, extra))
        return code_g

    def decode_pw(self, P):

        assert (
            len(P) == hny_config.PASSWORD_LENGTH
        ), "Not correct length to decode, Expecting {}, got {}".format(
            hny_config.PASSWORD_LENGTH, len(P)
        )

        iterp = iter(P)
        plaintext = ""
        stack = ["G"]
        while stack:
            lhs = stack.pop()
            rhs = self.decode_rule(lhs, next(iterp))

            if lhs in ["G", "T", "W", "Y", "D"]:  # 其实 'W', 'Y', 'D' 没有用
                arr = (
                    rhs.split(",")
                    if lhs != "T"
                    else ["T_%s" % c for c in rhs.split(",")]
                )
                arr.reverse()
                stack.extend(arr)
            elif lhs.startswith("W"):  # 碰到W,当然要动用 self.decode_l33t(rhs, iterp)
                rhs = self.decode_l33t(rhs, iterp)
                plaintext += rhs
            else:
                plaintext += rhs
        return plaintext

    # 增加了一个解码path的函数，方便使用
    def decode_l33t_path(self, w, iterp):
        # print("L33t:::", w, iterp)
        l = self.decode_rule("L", next(iterp))
        if l == "Caps":
            return [("L", "Caps")]
        elif l == "lower":
            return [("L", "lower")]
        elif l == "UPPER":
            return [("L", "UPPER")]
        else:
            leet_set = []
            leet_set.append(("L", "l33t"))
            for c in w:
                leet_set.append(("L_%s" % c, self.decode_rule("L_%s" % c, next(iterp))))
            return leet_set

    def decode_path(self, P):
        rule_set = []

        assert (
            len(P) == hny_config.PASSWORD_LENGTH
        ), "Not correct length to decode, Expecting {}, got {}".format(
            hny_config.PASSWORD_LENGTH, len(P)
        )

        iterp = iter(P)

        stack = ["G"]
        while stack:
            lhs = stack.pop()
            rhs = self.decode_rule(lhs, next(iterp))
            rule_set.append((lhs, rhs))
            if lhs in ["G", "T"]:
                arr = (
                    rhs.split(",")
                    if lhs != "T"
                    else ["T_%s" % c for c in rhs.split(",")]
                )
                arr.reverse()
                stack.extend(arr)
            elif lhs.startswith("W"):
                rhs = self.decode_l33t_path(rhs, iterp)
                rule_set.extend(rhs)
        return rule_set

    def codes2rules(self, codes):
        return self.decode_path(codes)

    def get_grammar(self, sg):
        stack = ["G"]
        sg_lst = []
        while stack:
            head = stack.pop()
            rule_dict = sg[head]
            reg_stack = []
            for rhs in rule_dict.keys():
                if rhs != "__total__":
                    r = [
                        x
                        for x in self.get_actual_NonTlist(head, rhs)
                        if x not in reg_stack and x not in ["W1", "D1", "Y1", "Y2"]
                    ]
                    for x in r:
                        reg_stack.append(x)
            reg_stack.reverse()
            stack.extend(reg_stack)
            n = len(list(rule_dict.keys())) - 1
            if n < 0:
                print(
                    "Sorry I cannot encode your password ({!r})! \nPlease choose"
                    " something different, like password12!! (Just kidding.)".format(
                        (head, list(rule_dict.keys()))
                    )
                )
                exit(0)
            sg_lst.append(rule_dict)
        return sg_lst

    def encode_grammar(self, G):  # G是一个 subGrammar类
        """
        Encodes a sub-grammar @G under the current grammar.
        Note: Does not record the frequencies.
        G->[
        """
        vd = VaultDistPCFG()
        stack = ["G"]
        code_g = []
        while stack:
            head = stack.pop()
            rule_dict = G[head]
            reg_stack = []
            for rhs in rule_dict.keys():
                if rhs != "__total__":
                    r = [
                        x
                        for x in self.get_actual_NonTlist(head, rhs)
                        if x not in reg_stack and x not in ["W1", "D1", "Y1", "Y2"]
                    ]
                    for x in r:
                        reg_stack.append(x)
            reg_stack.reverse()
            stack.extend(reg_stack)
            n = len(list(rule_dict.keys())) - 1
            code_g.append(vd.encode_vault_size(head, n))

            if n < 0:
                print(
                    "Sorry I cannot encode your password ({!r})! \nPlease choose"
                    " something different, like password12!! (Just kidding.)".format(
                        (head, list(rule_dict.keys()))
                    )
                )
                exit(0)
            assert n == vd.decode_vault_size(head, code_g[-1]), (
                "Vault size encoding mismatch.\nhead: {!r}, code_g: {}, n: {}, "
                "decoded_vault_size: {}".format(
                    head, code_g[-1], n, vd.decode_vault_size(head, code_g[-1])
                )
            )

            code_g.extend(
                [
                    self.encode_rule(head, r)
                    for r in rule_dict.keys()
                    if r != "__total__"
                ]
            )

        extra = hny_config.HONEY_VAULT_GRAMMAR_SIZE - len(code_g)
        code_g.extend(convert2group(0, 1, extra))
        return code_g

    def decode_grammar(self, P, TG):  # SG就是 一个SubGrammar类的对象
        """
        Decodes a subgrammar under self.G using the random numbers from P.
        """
        g = SubGrammar(base_pcfg=None, TG=TG)

        vd = VaultDistPCFG()

        iterp = iter(P)
        stack = ["G"]
        while stack:
            head = stack.pop()
            p = next(iterp)
            n = vd.decode_vault_size(head, p)
            reg_stack = []

            for _ in range(n):
                p = next(iterp)
                rhs = self.decode_rule(head, p)
                if rhs != "__totoal__":
                    r = [
                        y
                        for y in self.get_actual_NonTlist(head, rhs)
                        if y not in reg_stack and y not in ["W1", "D1", "Y1", "Y2"]
                    ]
                    for y in r:
                        reg_stack.append(y)
                else:
                    print(
                        ">>>>> __total__ should not be in the encoded grammar. Something is wrong!"
                    )
                g.add_rule(head, rhs)
            reg_stack.reverse()
            stack.extend(reg_stack)
        g.finalize()  # fixes the freq and some other book keepings
        return g

    def nonterminals(self):
        return list(self.G.keys())

    def __getitem__(self, l):
        return self.G[l]


MAX_ALLOWED = 20  # per rule


class VaultDistPCFG:
    def __init__(self):
        self.G = json.load(
            open_(hny_config.VAULT_DIST_FILE), object_pairs_hook=OrderedDict
        )
        # Add dummy entries for new non-terminals now
        # TODO: Learn them by vault analysis.
        # uniformly distribute these values between 1 and 30
        use_ful = 5
        for k in ["W", "D", "Y"]:  # 'R', 'T'
            self.G[k] = OrderedDict(
                zip(
                    (str(x + 1) for x in range(MAX_ALLOWED + 1)),
                    [100] * use_ful + [5] * (MAX_ALLOWED - use_ful),
                )
            )

        for k, v in list(self.G.items()):
            v["__total__"] = sum(v.values())

    def encode_vault_size(self, lhs, n):
        v = self.G.get(lhs, {})
        n = str(n)
        try:
            i = list(v.keys()).index(n)
            t_v = list(v.values())
            x = sum(t_v[:i])
            y = x + t_v[i]
            return convert2group(random.randint(x, y - 1), v["__total__"])
        except ValueError:
            return convert2group(0, v["__total__"])

    def decode_vault_size(self, lhs, cf):
        assert not lhs.startswith("L")
        assert (
            lhs in self.G
        ), "lhs={} must be in G. I cannot find it in\nG.keys()={}".format(
            lhs, list(self.G.keys())
        )
        cf %= self.G[lhs]["__total__"]
        if cf == 0:
            print_err(
                "Grammar of size 0!!!!\nI don't think the decryption will "
                "be right after this. I am sorry. Argument: (lhs: {}, cf: {})".format(
                    lhs, cf
                )
            )
        i = getIndex(cf, list(self.G[lhs].values()))
        return i + 1


class SubGrammar(TrainedGrammar):
    # 要不我们把L规则全部抄写上去！
    def __init__(self, base_pcfg=None, TG=None):  # base_pcfg就是 sub_tg
        self.tg = TG
        """
        odict_keys(['D1', 'D2', 'D3', 'D4', 'D5', 
        'G', 
        'L', 
        'L_a', 'L_b', 'L_c', 'L_d', 'L_e', 'L_f', 'L_g', 'L_h', 'L_i', 'L_j', 'L_k', 'L_l', 'L_m', 'L_n', 'L_o', 'L_p', 'L_q', 'L_r', 'L_s', 'L_t', 'L_u', 'L_v', 'L_w', 'L_x', 'L_y', 'L_z', 
        'R', 
        'T', 
        'T_Y', 'T_d', 'T_m', 'T_y', 
        'W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W9', 
        'Y1', 'Y2'])
        """
        self.cal_cdf = False

        R = RuleSet()

        self.base_pcfg = base_pcfg

        default_keys = []
        no_consider_keys = []
        consider_keys = []

        R.update_set(RuleSet(d={"L": self.tg.G["L"]}))
        default_keys.append("L")
        no_consider_keys.append("L")

        for c in string.ascii_lowercase:  # L_*
            x = "L_%s" % c  # 假设这里好像有问题
            default_keys.append(x)
            no_consider_keys.append(x)
            # {L_a:{a:12,A:15,@:16}}
            R.update_set(RuleSet(d={x: self.tg.G[x]}))

        for k in ["W1", "D1", "Y1", "Y2"]:
            # 例如 k = W1,G  v = 162
            default_keys.append(k)
            no_consider_keys.append(k)

            R.update_set(RuleSet(d={k: self.tg.G[k]}))

        # for num# 189 pwd:f92f45,必须增加G-D1
        R.update_set(
            RuleSet(
                d={"G": {"D1": 2, "W1": 2, "Y1": 2, "W1,G": 2, "D1,G": 2, "Y1,G": 2}}
            )
        )

        if self.base_pcfg:  # 有self.base_pcfg存在的条件下，self.base_pcfg是一个TG类
            default_keys.append("G")
            consider_keys.append("G")

            for it in self.base_pcfg.G["G"].items():
                if it[0] not in ["W1", "D1", "Y1", "D1,G", "Y1,G", "W1,G"]:
                    R.update_set(RuleSet(d={"G": {it[0]: it[1]}}))

            for nont_key in self.base_pcfg.G.keys():
                if nont_key not in R.G.keys():
                    default_keys.append(nont_key)
                    consider_keys.append(nont_key)
                    R.update_set(RuleSet(d={nont_key: self.base_pcfg.G[nont_key]}))

        self._default_keys = set(default_keys)
        self._no_consider_keys = set(no_consider_keys)
        self._consider_keys = set(consider_keys)
        self.R = R
        self.G = R.G  # R只是一个对象，R.G才是最终发挥功能的那个属性字典

        self.date = Date()
        self.freeze = False

    def add_rule(self, l, r):
        if self.freeze:
            print("Warning! Please defreeze the grammar before adding")
        self.R.add_rule(l, r)

    def finalize(self):  # 这个函数运行后，最新的R.G给到了sub_tg.G, 功能开始发挥
        self.fix_freq()
        self.NonT_set = [
            x for x in list(self.G.keys()) if x.find("_") < 0
        ]  # + list('Yymd')
        self.G = self.R.G

        Wlist = [
            x
            for k, v in list(self.G.items())
            for x in v
            if k.startswith("W") and x != "__total__"
        ]
        self.Wdawg = IntDAWG(Wlist)

        for k, v in self.G.items():

            for rhs, f in v.items():
                if f <= 0:
                    print("Zero frequency LHS added, setting frequency to 1")
                    v[rhs] = 1
                    if "__total__" in v:
                        v["__total__"] += 1
            if "__total__" not in v:
                print("__total__ should be there in the keys!!. I am adding one.")
                v["__total__"] = sum(v.values())

        if "T" in self.G:
            self.date = Date(
                T_rules=[x for x in list(self.G["T"].keys()) if x != "__total__"]
            )
        self.freeze = True
        self.R.G.default_factory = None

    def reset(self):
        for k, v in list(self.G.items()):
            if "__total__" in v:
                del v["__total__"]
        self.freeze = False
        self.R.G.default_factory = OrderedDict

    def add_some_extra_rules(self):
        for k, v in list(self.R.items()):
            pass

    def update_grammar(self, *args):
        self.reset()
        for pw in args:
            pw = pw.replace("\\", "")
            self.R.update_set(self.base_pcfg.rule_set(pw))
        self.finalize()

    def fix_freq(self):
        for l, v in list(self.R.items()):
            s = 0
            for r in v:
                if r != "__total__":
                    if r not in ["D1", "W1", "Y1", "D1,G", "W1,G", "Y1,G"]:
                        v[r] = self.tg.get_freq(l, r)
                    else:
                        v[r] = 2
                    s += v[r]
            v["__total__"] = s

    def prob(self, pw, selection):
        pt = self.l_parse_tree(pw, selection)
        proba = 1.0
        for it in pt:
            proba *= self.get_prob(it[0], it[1])
        if proba == 0:
            proba = 1e-10
        return proba

    def pw2rules(self, word):
        return self.l_parse_tree(word)

    def check(self, pt1, pt2):
        if len(pt1) != len(pt2):
            return 0
        else:
            len_pt = len(pt1)
            for i in range(len_pt):
                if pt1[i][0] != pt2[i][0] or pt1[i][1] != pt2[i][1]:
                    return 0
            return 1

    def codes2rules(self, codes):
        return self.decode_path(codes)

    def default_keys(self):
        return self._default_keys

    def no_consider_keys(self):
        return self._no_consider_keys

    def consider_keys(self):
        return self._consider_keys


def make_subgrammar(list, tg, selection):
    rule_cumu = []
    for it in list:
        pt = tg.l_parse_tree(it, selection)
        rule_cumu.extend(pt)

    sub_grammar_cumu = {}  # 从rule到grammar需要进一步操作
    for it in rule_cumu:
        k = it[0]
        v = it[1]
        if k not in sub_grammar_cumu.keys():
            sub_grammar_cumu[k] = {}
        if v not in sub_grammar_cumu[k].keys():
            val = tg.get_freq(k, v)
            sub_grammar_cumu[k][v] = val

    small_tg = TrainedGrammar(g_file="xxx", g_itself=sub_grammar_cumu)
    sub_tg = SubGrammar(base_pcfg=small_tg, TG=tg)
    sub_tg.finalize()
    return sub_tg
