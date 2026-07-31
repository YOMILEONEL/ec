#!/usr/bin/env python3
"""DeepCoder-tailored DreamCoder primitive set, used by step04_run_dreamcoder_train_test.py.

dreamcoder.domains.list.listPrimitives.primitives() is deliberately minimal
(map and fold are commented out, there is no zip) because it is the
"bootstrapping" set from the original DreamCoder paper: DreamCoder is meant to
*rediscover* map/fold/zip from smaller pieces via library learning over many
iterations. With grammar consolidation disabled (as most runs in this pipeline
do, to avoid depending on the Rust compressor) or with only a couple of short
iterations, that rediscovery essentially never happens, so the vast majority
of tasks in this dataset (which need map/filter/zip directly, since the
DeepCoder DSL is built on exactly those combinators) are unsolvable regardless
of training-set size or search timeout. This module hands DreamCoder those
combinators directly instead.

This lives in its own real, importable module (not inside the dynamically
generated __main__ runner script) on purpose: DreamCoder's parallel solver
uses dill/multiprocess to move enumerated programs between worker processes,
which reconstructs functions defined in a throwaway __main__ script as new,
non-identical objects. Plain pickle.dumps() of the final result then fails
with "it's not the same object as __main__.<name>". Functions that live in a
real module are resolved consistently by import path in every process, the
same way dreamcoder.domains.list.listPrimitives functions already are.
"""


def deepcoder_division(x):
    return lambda y: 0 if y == 0 else int(x / y)


def deepcoder_lt(x):
    return lambda y: x < y


def deepcoder_min2(x):
    return lambda y: x if x < y else y


def deepcoder_max2(x):
    return lambda y: x if x > y else y


def deepcoder_zipWith(f):
    return lambda a: lambda b: list(map(lambda x, y: f(x)(y), a, b))


def deepcoder_scanl1_run(f, l):
    if not l:
        return []
    acc = l[0]
    out = [acc]
    for x in l[1:]:
        acc = f(acc)(x)
        out.append(acc)
    return out


def deepcoder_scanl1(f):
    return lambda l: deepcoder_scanl1_run(f, l)


def build_deepcoder_primitives():
    from dreamcoder.program import Primitive
    from dreamcoder.type import tlist, tint, tbool, arrow, t0, t1, t2
    from dreamcoder.domains.list.listPrimitives import (
        _map, _filter, _fold, _index, _reverse, _negate, _mod, _gt, _eq,
        _addition, _subtraction, _multiplication, _any, _all, _isEmpty,
        _cons, _car, _cdr, _if, _single, _append, _slice, _isPrime, _isSquare,
    )

    return [Primitive(str(j), tint, j) for j in range(6)] + [
        Primitive("empty", tlist(t0), []),
        Primitive("singleton", arrow(t0, tlist(t0)), _single),
        Primitive("++", arrow(tlist(t0), tlist(t0), tlist(t0)), _append),
        Primitive("cons", arrow(t0, tlist(t0), tlist(t0)), _cons),
        Primitive("car", arrow(tlist(t0), t0), _car),
        Primitive("cdr", arrow(tlist(t0), tlist(t0)), _cdr),
        Primitive("empty?", arrow(tlist(t0), tbool), _isEmpty),
        Primitive("if", arrow(tbool, t0, t0, t0), _if),
        Primitive("length", arrow(tlist(t0), tint), len),
        Primitive("reverse", arrow(tlist(t0), tlist(t0)), _reverse),
        Primitive("sort", arrow(tlist(tint), tlist(tint)), sorted),
        Primitive("sum", arrow(tlist(tint), tint), sum),
        Primitive("index", arrow(tint, tlist(t0), t0), _index),
        Primitive("slice", arrow(tint, tint, tlist(t0), tlist(t0)), _slice),
        Primitive("map", arrow(arrow(t0, t1), tlist(t0), tlist(t1)), _map),
        Primitive("filter", arrow(arrow(t0, tbool), tlist(t0), tlist(t0)), _filter),
        Primitive("fold", arrow(arrow(t1, t0, t1), t1, tlist(t0), t1), _fold),
        Primitive("zipWith", arrow(arrow(t0, t1, t2), tlist(t0), tlist(t1), tlist(t2)), deepcoder_zipWith),
        Primitive("scanl1", arrow(arrow(t0, t0, t0), tlist(t0), tlist(t0)), deepcoder_scanl1),
        Primitive("any", arrow(arrow(t0, tbool), tlist(t0), tbool), _any),
        Primitive("all", arrow(arrow(t0, tbool), tlist(t0), tbool), _all),
        Primitive("+", arrow(tint, tint, tint), _addition),
        Primitive("-", arrow(tint, tint, tint), _subtraction),
        Primitive("*", arrow(tint, tint, tint), _multiplication),
        Primitive("/", arrow(tint, tint, tint), deepcoder_division),
        Primitive("mod", arrow(tint, tint, tint), _mod),
        Primitive("negate", arrow(tint, tint), _negate),
        Primitive("min", arrow(tint, tint, tint), deepcoder_min2),
        Primitive("max", arrow(tint, tint, tint), deepcoder_max2),
        Primitive("gt?", arrow(tint, tint, tbool), _gt),
        Primitive("lt?", arrow(tint, tint, tbool), deepcoder_lt),
        Primitive("eq?", arrow(tint, tint, tbool), _eq),
        Primitive("is-prime", arrow(tint, tbool), _isPrime),
        Primitive("is-square", arrow(tint, tbool), _isSquare),
    ]
