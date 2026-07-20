import sys
import inspect
from tamarl.rl.train_bandit import _build_parser, _CLI_TO_KWARGS, train

def test_argument_parser():
    parser = _build_parser()
    
    # Test case 1: without the flag
    args = parser.parse_args(["--scenario", "tamarl/data/scenarios/grid_world/3x3"])
    assert getattr(args, "render_no_text", None) is None
    
    # Test case 2: with the flag
    args_with = parser.parse_args(["--scenario", "tamarl/data/scenarios/grid_world/3x3", "--render_no_text"])
    assert getattr(args_with, "render_no_text") is True

def test_cli_to_kwargs():
    assert "render_no_text" in _CLI_TO_KWARGS
    assert _CLI_TO_KWARGS["render_no_text"] == "render_no_text"

def test_train_signature():
    sig = inspect.signature(train)
    assert "render_no_text" in sig.parameters
    assert sig.parameters["render_no_text"].default is False
