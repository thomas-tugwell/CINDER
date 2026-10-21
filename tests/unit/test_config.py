import pytest

from cinder.config import ConfigError, config_from_dict, load_config


def test_defaults_valid():
    cfg = config_from_dict({})
    assert cfg.conformation.n_samples == 8


def test_unknown_key_rejected():
    with pytest.raises(ConfigError, match="unknown config key"):
        config_from_dict({"conformation": {"n_sample": 3}})
    with pytest.raises(ConfigError):
        config_from_dict({"bogus": 1})


@pytest.mark.parametrize(
    "bad",
    [
        {"conformation": {"n_samples": 0}},
        {"optimization": {"learning_rate": -1}},
        {"sequence": {"init": "random"}},
        {"system": {"kind": "structure"}},
        {"energy": {"backend": "magic"}},
        {"energy": {"hbond_weight": -1.0}},
        {"energy": {"elec_weight": -0.1}},
    ],
)
def test_validation_errors(bad):
    with pytest.raises(ConfigError):
        config_from_dict(bad)


def test_defaults_fragments_and_overrides(tmp_path):
    (tmp_path / "frag").mkdir()
    (tmp_path / "frag" / "c.yaml").write_text("conformation:\n  n_samples: 3\n  n_steps: 10\n")
    exp = tmp_path / "exp"
    exp.mkdir()
    (exp / "e.yaml").write_text("defaults: [frag/c]\nconformation:\n  n_steps: 20\n")
    cfg = load_config(exp / "e.yaml", {"seed": 7})
    assert (cfg.conformation.n_samples, cfg.conformation.n_steps, cfg.seed) == (3, 20, 7)


def test_repo_configs_load():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2] / "configs" / "experiment"
    for p in root.glob("[!._]*.yaml"):
        load_config(p)


def test_hbond_graph_options_are_validated():
    from cinder.energy.tmol_design import graph_config_from_options

    cfg = graph_config_from_options({"hbond_graph": {"w_second": 2.0}})
    assert cfg.w_second == 2.0 and cfg.w_first == 1.0
    with pytest.raises(ValueError, match="unknown"):
        graph_config_from_options({"hbond_graph": {"w_secnd": 1.0}})
    with pytest.raises(ConfigError):
        config_from_dict({"energy": {"hbond_graph_weight": -1.0}})
