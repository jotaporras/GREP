"""Cross-module contract for the undirected-edge ``u <=> v`` notation.

The ``<=>`` token is a shared format spanning three modules, and the contract is
that whatever one module *writes* the others can *read* — and that ``<=>`` edge
statements never collide with ``->`` route hops:

* ``data.compact_prompt._edges``  EMITS edges as ``u <=> v`` (the plain-LLM
  ``• Region Edges:`` / ``• Object Edges:`` bullets).
* ``eval.path_validator.parse_edges``  PARSES ``u <=> v`` back to frozenset pairs.
* ``eval.path_validator.parse_path``  reads ``->`` routes and must NOT mistake a
  ``<=>`` edge for a route hop (path_validator.py:91-94 collision-freedom claim).
* ``data.graph_gen.UPDATED_QUERY``  INSTRUCTS the generator LLM to state edges as
  ``A <=> B`` — so the example it embeds must be parseable by ``parse_edges``,
  else generated data is graded against a notation the grader can't read.

The oracle is independent of the implementations: edge sets are built by hand
from the input pairs, never by re-running the regex under test.

Existing suites already cover parse-side basics (``test_path_validator.py`` ::
``test_parse_edges_*`` / ``test_parse_path_neutralizes_undirected_edge_arrows``);
this file pins the EMIT side and the EMIT->PARSE round-trip those don't touch.
"""

import sys

sys.path.insert(0, "src")

from prism.data.compact_prompt import _edges
from prism.eval.path_validator import parse_edges, parse_path


def _pset(*pairs):
    """Hand-built oracle: a set of frozenset edges from (u, v) tuples."""
    return {frozenset(p) for p in pairs}


# --------------------------------------------------------------------------
# Emit side: compact_prompt._edges
# --------------------------------------------------------------------------
def test_edges_emits_double_arrow_notation():
    """``_edges`` joins ``[u, v]`` pairs as ``u <=> v``, comma-separated."""
    out = _edges([["hub_1", "mess_hall_1"], ["mess_hall_1", "food_dispenser_1"]])
    assert out == "hub_1 <=> mess_hall_1, mess_hall_1 <=> food_dispenser_1"


def test_edges_skips_non_pair_entries():
    """Malformed (non-2) entries are dropped, not raised on (docstring contract)."""
    out = _edges([["a_1", "b_1"], ["c_1"], ["d_1", "e_1", "f_1"], ["g_1", "h_1"]])
    assert out == "a_1 <=> b_1, g_1 <=> h_1"


def test_edges_empty_list_is_empty_string():
    assert _edges([]) == ""


# --------------------------------------------------------------------------
# Emit -> parse round-trip: _edges output must be recoverable by parse_edges
# --------------------------------------------------------------------------
def test_emit_parse_roundtrip_recovers_exact_edges():
    """What ``_edges`` writes, ``parse_edges`` reads back — same undirected pairs.

    This is the load-bearing cross-module claim: the plain-LLM text block format
    and the grader's edge parser agree on ``<=>``.
    """
    pairs = [["hub_1", "mess_hall_1"], ["mess_hall_1", "food_dispenser_1"],
             ["bay_3_26_1", "dock_2"]]  # multi-tail grid id included
    text = _edges(pairs)
    assert parse_edges(text) == _pset(*pairs)


def test_parse_edges_tolerates_spacing_around_double_arrow():
    """The ``\\s*<=>\\s*`` regex accepts zero or many spaces, not just the single
    space ``_edges`` emits (models reproduce the token with varied spacing)."""
    assert parse_edges("hub_1<=>mess_hall_1") == _pset(("hub_1", "mess_hall_1"))
    assert parse_edges("hub_1   <=>   mess_hall_1") == _pset(("hub_1", "mess_hall_1"))


# --------------------------------------------------------------------------
# Non-collision: <=> (edges) vs -> (routes) read disjoint from one mixed text
# --------------------------------------------------------------------------
def test_edges_and_route_parsed_disjointly_from_mixed_text():
    """In a response carrying BOTH an edge list and a route, ``parse_edges`` reads
    only the ``<=>`` pairs and ``parse_path`` reads only the ``->`` hops — neither
    bleeds into the other (path_validator.py:91-94)."""
    text = ("Edges: hub_1 <=> mess_hall_1. "
            "Route: hub_1 -> mess_hall_1 -> food_dispenser_1")
    # Only the one stated <=> edge — the route's second hop is NOT an edge here.
    assert parse_edges(text) == _pset(("hub_1", "mess_hall_1"))
    # The full 3-node route — unaffected by the <=> edge statement.
    assert parse_path(text) == ["hub_1", "mess_hall_1", "food_dispenser_1"]


def test_double_arrow_breaks_an_arrow_chain():
    """A ``<=>`` sitting between two nodes is not a hop connector: it severs the
    ``->`` chain rather than fusing the four nodes into one route."""
    text = "hub_1 -> mess_hall_1 <=> food_dispenser_1 -> store_1"
    nodes = parse_path(text)
    # The <=> must NOT have been read as a connector producing a 4-node walk.
    assert "food_dispenser_1" not in nodes or nodes[: nodes.index("food_dispenser_1")] != [
        "hub_1", "mess_hall_1"]
    assert nodes == ["hub_1", "mess_hall_1"]


# --------------------------------------------------------------------------
# Cross-file: the notation graph_gen instructs the LLM to emit must parse
# --------------------------------------------------------------------------
def test_graph_gen_prompt_edge_example_is_parseable():
    """The ``A <=> B`` example baked into ``UPDATED_QUERY`` must be exactly the
    form ``parse_edges`` recognises — otherwise the generator is told to produce a
    notation the deterministic grader cannot read.

    ``graph_gen`` imports the heavy spine/torch stack at module load; skip if that
    stack is unavailable here (the ``<=>`` constant is the only thing under test).
    """
    try:
        from prism.data import graph_gen
    except Exception as e:  # noqa: BLE001 — heavy optional deps (spine / torch_geometric)
        import os
        msg = f"graph_gen import unavailable: {type(e).__name__}: {e}"
        if "PYTEST_CURRENT_TEST" in os.environ:
            import pytest
            pytest.skip(msg)
        print(f"[SKIP] {msg}")
        return

    prompt = graph_gen.UPDATED_QUERY
    parsed = parse_edges(prompt)
    # The criterion example states `fuel_depot_1 <=> fuel_tank_1`.
    assert frozenset(("fuel_depot_1", "fuel_tank_1")) in parsed, (
        "graph_gen's instructed <=> example is not parseable by path_validator.parse_edges")
    # The example *route* uses `->`, written as `clearing_1 -> comm_bunker_1 -> ...`;
    # that hop is never stated as a `<=>`/`[..]`/`(..)` pair, so it must NOT surface
    # as an edge — confirming `->` route notation does not leak into the edge parser.
    assert frozenset(("clearing_1", "comm_bunker_1")) not in parsed, (
        "a `->` route hop leaked into parse_edges as an undirected edge")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"{name}: PASS")
    print("done")
