"""Score barnase/barstar with TMol and print the design-energy components (needs TMol)."""

from cinder.config import load_config
from cinder.experiments.runner import score_native

print(score_native(load_config("configs/experiment/score.yaml")))
