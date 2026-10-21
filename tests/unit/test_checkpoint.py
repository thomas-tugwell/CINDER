import torch

from cinder.experiments.checkpoint import load_checkpoint, save_checkpoint
from cinder.sequence import AutoregressivePolicy, CategoricalPolicy
from cinder.types import N_AA


def test_checkpoint_round_trip_restores_policy_and_optimizer(tmp_path):
    mask = torch.ones(3, N_AA, dtype=torch.bool)
    for make in (CategoricalPolicy, AutoregressivePolicy):
        policy = make(mask)
        opt = torch.optim.Adam(policy.parameters(), lr=0.1)
        policy.sample(8).log_prob.sum().backward()
        opt.step()
        path = tmp_path / f"{make.__name__}.pt"
        save_checkpoint(path, policy, opt, iteration=7, extra={"best_sequence": "ACD"})

        fresh = make(mask)
        fresh_opt = torch.optim.Adam(fresh.parameters(), lr=0.1)
        ckpt = load_checkpoint(path, fresh, fresh_opt)
        assert ckpt["iteration"] == 7 and ckpt["extra"]["best_sequence"] == "ACD"
        for a, b in zip(policy.parameters(), fresh.parameters(), strict=True):
            assert torch.equal(a, b)
        assert len(fresh_opt.state_dict()["state"]) == len(opt.state_dict()["state"])
