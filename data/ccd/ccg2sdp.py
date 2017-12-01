#!/usr/bin/env python3

# Convert CCGBank to SDP 2015 format
# Marco Kuhlmann <marco.kuhlmann@liu.se>

# Usage: ccg2sdp.py 2005T13/data > ccg.sdp

import os
import re

def parse_bracket(line, beg, end):
    assert line.startswith(beg)
    depth = 0
    for i in range(len(line)):
        if line[i] == beg:
            depth += 1
        if line[i] == end:
            depth -= 1
            if depth == 0:
                return line[1:i].strip(), line[i+1:].lstrip()
    assert False

def parse_tree(line):
    root, rest = parse_bracket(line, '<', '>')
    root = root.split()
    if root[0] == "L":
        # CCGcat mod_POS-tag orig_POS-tag word PredArgCat
        return (root[1:], [])
    if root[0] == "T":
        # CCGCat head dtrs
        root[2] = int(root[2])
        root[3] = int(root[3])
        subtrees = []
        for _ in range(root[3]):
            subtree, rest = parse_bracket(rest, '(', ')')
            subtrees.append(parse_tree(subtree))
        assert len(subtrees) > 0
        return (root[1:], subtrees)
    assert False

def parse(line):
    tree, rest = parse_bracket(line.lstrip(), '(', ')')
    assert len(rest) == 0
    return parse_tree(tree)

def collect_tokens(tree, is_top):
    root, subtrees = tree
    if len(subtrees) == 0:
        # CCGCat mod_POS-tag orig_POS-tag word PredArgCat IsTop
        yield root + [is_top]
    else:
        for i, subtree in enumerate(subtrees):
            for token in collect_tokens(subtree, is_top and root[1] == i):
                yield token

ID = "([0-9]{2})([0-9]{2})\.([1-9][0-9]*)"

def get_instances_toks(fp):
    for line in fp:
        line = line.rstrip()
        if line.startswith("ID"):
            ids = re.match(r'ID=wsj_%s' % ID, line)
            assert ids is not None
            ids = [int(x) for x in ids.groups()]
        else:
            tokens = list(collect_tokens(parse(line), True))
            yield (ids, tokens)

def get_instances_deps(fp):
    for line in fp:
        line = line.rstrip()
        if line.startswith("<s id="):
            match = re.match(r'<s id="wsj_%s">' % ID, line)
            assert match is not None
            match = [int(x) for x in match.groups()]
            edges = set()
        elif line.startswith("<\s>"):
            yield (match, edges)
        else:
            columns = line.split()
            tgt = int(columns[0])
            src = int(columns[1])
            cat = columns[2]
            arg = int(columns[3])
            edges.add((src, tgt, cat, arg))

def get_instances(datadir):
    autodir = os.path.join(datadir, "AUTO")
    pargdir = os.path.join(datadir, "PARG")
    for auto, parg in zip(os.walk(autodir), os.walk(pargdir)):
        for auto_file, parg_file in zip(auto[2], parg[2]):
            auto_fp = open(os.path.join(auto[0], auto_file))
            parg_fp = open(os.path.join(parg[0], parg_file))
            instances_toks = get_instances_toks(auto_fp)
            instances_deps = get_instances_deps(parg_fp)
            for i_toks, i_deps in zip(instances_toks, instances_deps):
                assert i_toks[0] == i_deps[0]
                # sentence_id tokens dependencies
                yield (i_toks[0], i_toks[1], i_deps[1])
            auto_fp.close()
            parg_fp.close()

def emit_instance_as_sdp(instance):
    sentence_id, tokens, dependencies = instance

    sec, doc, sen = sentence_id
    print("#2%02d%02d%03d" % (sec, doc, sen))

    predicates = set()
    roles = {}
    for src, tgt, cat, arg in dependencies:
        predicates.add(src)
        roles[(src, tgt)] = arg
        assert cat == tokens[src][0]
    predicates = sorted(predicates)
    
    for tgt, token in enumerate(tokens):
        row = []
        row.append(str(tgt+1))
        row.append(token[3]) # word
        row.append("_") # (no lemma information)
        row.append(token[1]) # orig_POS-tag
        row.append("+" if token[5] else "-")
        row.append("+" if tgt in predicates else "-")
        row.append(token[0]) # CCGcat
        for src in predicates:
            if (src, tgt) in roles:
                row.append("%d" % roles[(src, tgt)])
            else:
                row.append("_")
        print("\t".join(row))
    print()

def emit_as_sdp(datadir, r):
    print("#SDP 2015")
    for instance in get_instances(datadir):
        sec = instance[0][0]
        if sec in r:
            emit_instance_as_sdp(instance)

if __name__ == "__main__":
    import sys
    sections = range(00, 25) # all sections
    if len(sys.argv) > 1:
        if sys.argv[2] == "training":
            sections = range(2, 22)
        if sys.argv[2] == "development":
            sections = range(0, 1)
        if sys.argv[2] == "testing":
            sections = range(23, 24)
    emit_as_sdp(sys.argv[1], sections)
