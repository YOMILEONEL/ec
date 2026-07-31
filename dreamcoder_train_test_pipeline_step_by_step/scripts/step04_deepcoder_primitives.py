#!/usr/bin/env python3
"""DeepCoder-tailored DreamCoder primitive set, used by step04_run_dreamcoder_train_test.py.

dreamcoder.domains.list.listPrimitives.primitives() is deliberately minimal
(map and fold are commented out, there is no zip) because it is the
"bootstrapping" set from the original DreamCoder paper: DreamCoder is meant to
*rediscover* map/fold/zip from smaller pieces via library learning over many
iterations. That rediscovery needs grammar consolidation (compression) to
actually run many iterations with a working compressor - without it, or with
only a couple of short iterations, it essentially never happens, so the vast
majority of tasks in this dataset (which need map/filter/zip directly, since
the DeepCoder DSL is built on exactly those combinators) are unsolvable
regardless of training-set size or search timeout. This module hands
DreamCoder those combinators directly instead.

Every primitive below is chosen to match a name AND type signature that is
already hardcoded into the OCaml compressor binary (solvers/program.ml). That
binary has a fixed, compiled-in table of primitive names it can parse; a
primitive whose name isn't in that table (e.g. a hand-rolled "sum", "min",
"max", "lt?", "/", "zipWith", "scanl1") makes consolidation crash with
"safe_get_some failure: Error parsing: <name>" the moment a training task
using it gets solved, because compression.ml can no longer reconstruct the
program AST it was sent. Primitives DreamCoder still needs but that have no
OCaml counterpart (sum, min, max, division, running-scan) are intentionally
left out here; they are still reachable by composition from what remains
(e.g. sum = fold(+, 0, ...), max = fold(if gt?, ...)) and are exactly the
kind of reusable pattern that library learning (consolidation) is supposed to
discover and name for itself, rather than being handed as a primitive.

This lives in its own real, importable module (not inside the dynamically
generated __main__ runner script) on purpose: DreamCoder's parallel solver
uses dill/multiprocess to move enumerated programs between worker processes,
which reconstructs functions defined in a throwaway __main__ script as new,
non-identical objects. Plain pickle.dumps() of the final result then fails
with "it's not the same object as __main__.<name>". Functions that live in a
real module are resolved consistently by import path in every process, the
same way dreamcoder.domains.list.listPrimitives functions already are.
"""


def build_deepcoder_primitives():
    from dreamcoder.program import Primitive
    from dreamcoder.type import tlist, tint, tbool, arrow, t0, t1, t2
    from dreamcoder.domains.list.listPrimitives import (
        _map, _filter, _fold, _index, _reverse, _negate, _mod, _gt, _eq,
        _addition, _subtraction, _multiplication, _isEmpty,
        _cons, _car, _cdr, _if, _single, _append, _slice, _isPrime, _isSquare,
        _mapi, _reducei, _zip,
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
        Primitive("true", tbool, True),
        Primitive("length", arrow(tlist(t0), tint), len),
        Primitive("reverse", arrow(tlist(t0), tlist(t0)), _reverse),
        Primitive("sort", arrow(tlist(tint), tlist(tint)), sorted),
        Primitive("index", arrow(tint, tlist(t0), t0), _index),
        Primitive("slice", arrow(tint, tint, tlist(t0), tlist(t0)), _slice),
        Primitive("map", arrow(arrow(t0, t1), tlist(t0), tlist(t1)), _map),
        Primitive("mapi", arrow(arrow(tint, t0, t1), tlist(t0), tlist(t1)), _mapi),
        Primitive("filter", arrow(arrow(t0, tbool), tlist(t0), tlist(t0)), _filter),
        Primitive("fold", arrow(tlist(t0), t1, arrow(t0, t1, t1), t1), _fold),
        Primitive("reducei", arrow(arrow(tint, t1, t0, t1), t1, tlist(t0), t1), _reducei),
        Primitive("zip", arrow(tlist(t0), tlist(t1), arrow(t0, t1, t2), tlist(t2)), _zip),
        Primitive("+", arrow(tint, tint, tint), _addition),
        Primitive("-", arrow(tint, tint, tint), _subtraction),
        Primitive("*", arrow(tint, tint, tint), _multiplication),
        Primitive("mod", arrow(tint, tint, tint), _mod),
        Primitive("negate", arrow(tint, tint), _negate),
        Primitive("gt?", arrow(tint, tint, tbool), _gt),
        Primitive("eq?", arrow(tint, tint, tbool), _eq),
        Primitive("is-prime", arrow(tint, tbool), _isPrime),
        Primitive("is-square", arrow(tint, tbool), _isSquare),
    ]
