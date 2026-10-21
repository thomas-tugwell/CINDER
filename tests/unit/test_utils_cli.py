import torch

from cinder.cli import main
from cinder.doctor import run_doctor
from cinder.utils.seed import make_generator, seed_everything


def test_seeding_reproducible():
    seed_everything(5)
    a = torch.randn(3)
    seed_everything(5)
    assert torch.equal(a, torch.randn(3))
    g1, g2 = make_generator(1), make_generator(1)
    assert torch.equal(torch.randn(3, generator=g1), torch.randn(3, generator=g2))


def test_doctor_report_keys():
    r = run_doctor()
    assert {"python", "torch", "tmol", "bioemu"} <= set(r)


def test_cli_doctor_score_sample_design_analyze(tmp_path, capsys):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        f"output_dir: {tmp_path}/runs\nsystem: {{binder_length: 2}}\n"
        "conformation: {n_samples: 4, n_steps: 5}\noptimization: {iterations: 3, batch_size: 4}\n"
    )
    assert main(["doctor"]) == 0
    assert main(["score", "--config", str(cfg)]) == 0
    assert main(["sample", "--config", str(cfg)]) == 0
    assert main(["design", "--config", str(cfg)]) == 0
    run = next((tmp_path / "runs").iterdir())
    assert main(["analyze", str(run)]) == 0
    out = capsys.readouterr().out
    assert "best sequence" in out


def test_cli_config_error_exit_code(tmp_path):
    cfg = tmp_path / "bad.yaml"
    cfg.write_text("nonsense: 1\n")
    assert main(["score", "--config", str(cfg)]) == 2


def test_device_resolution_prefers_gpu_only_when_asked_and_available(monkeypatch):
    from cinder.energy.tmol import tmol_device
    from cinder.utils.device import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu" and tmol_device("auto").type == "cpu"
    assert tmol_device("cuda").type == "cpu"  # asked for CUDA, none present -> CPU
    assert tmol_device("cpu").type == "cpu" and tmol_device("mps").type == "cpu"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto").type == "cuda" and tmol_device("auto").type == "cuda"
    assert tmol_device("cuda:1") == torch.device("cuda:1")
    assert tmol_device("cpu").type == "cpu"  # explicit CPU is respected even with a GPU

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto").type == "mps"  # toy path may use MPS
    assert tmol_device("auto").type == "cpu"  # TMol has no MPS backend
