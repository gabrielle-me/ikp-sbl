from modules.OLD_search_tree import BidirectionalSBL, SearchTree
from lecture_examples.IPEnvironment import CollisionChecker


def test_grow_trees_handles_no_new_node(monkeypatch):
    checker = CollisionChecker({})
    planner = BidirectionalSBL(checker)

    monkeypatch.setattr(planner, "_expand_tree", lambda tree, config: None)

    tree_start, tree_goal, path = planner.grow_trees(
        [1.0, 1.0],
        [20.0, 20.0],
        {"max_nodes": 3, "eta": 1.0},
    )

    assert tree_start is not None
    assert tree_goal is not None
    assert path is None


def test_try_connect_adds_unknown_bridge_edge():
    checker = CollisionChecker({})
    planner = BidirectionalSBL(checker)

    active_tree = SearchTree([0.0, 0.0])
    active_tree.add_node([1.0, 0.0], parent=0)

    passive_tree = SearchTree([10.0, 10.0])
    passive_tree.add_node([11.0, 11.0], parent=0)
    passive_tree.add_node([12.0, 12.0], parent=1)

    passive_tree.nearest = lambda position: (2, 0.0)

    connection = planner._try_connect(active_tree, passive_tree, 1, {"eta": 10.0})

    assert connection is not None
    assert 2 in active_tree.graph.nodes
    assert active_tree.graph.nodes[2]["pos"] == [12.0, 12.0]
    assert active_tree.graph.edges[1, 2]["status"] == "unknown"
