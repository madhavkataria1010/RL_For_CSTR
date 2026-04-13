"""DR-CIRL training with frozen domain-randomization hooks."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.controllers.dr_cirl import DRCIRLPolicy
from src.training.common_interface import build_paper_training_payload, load_project_config
from src.training.train_cirl import (
    PaperTrainingConfig,
    RandomSearchPSOTrainer,
    ensure_run_dir,
    load_yaml_config,
    save_training_outputs,
)


def train_dr_cirl(
    config: PaperTrainingConfig | Mapping[str, Any] | Sequence[str | Path] | str | Path | None = None,
    **overrides: Any,
) -> dict[str, str]:
    if isinstance(config, Sequence) and not isinstance(config, (str, Path, bytes)):
        merged = load_project_config([str(path) for path in config])
        cfg = PaperTrainingConfig.from_mapping(
            build_paper_training_payload(merged, default_method_id="dr_cirl"),
            **overrides,
        )
    elif isinstance(config, (str, Path)):
        cfg = PaperTrainingConfig.from_mapping(load_yaml_config(config), **overrides)
    elif isinstance(config, PaperTrainingConfig):
        cfg = PaperTrainingConfig.from_mapping(asdict(config), **overrides)
    else:
        cfg = PaperTrainingConfig.from_mapping(config, **overrides)

    cfg.method_id = "dr_cirl"
    cfg.output_size = 6
    if not cfg.env_overrides.get("domain_randomization"):
        cfg.env_overrides["domain_randomization"] = {
            "enabled": True,
            "ranges": {
                "caf_scale": 0.10,
                "ua_scale": 0.10,
                "k0_scale": 0.10,
            },
        }

    run_dir = ensure_run_dir(cfg.results_root, cfg.experiment_name, cfg.method_id)
    trainer = RandomSearchPSOTrainer(
        cfg,
        lambda: DRCIRLPolicy(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden_size,
            output_size=cfg.output_size,
            activation=getattr(torch.nn, cfg.activation),
            extra_hidden_layers=cfg.extra_hidden_layers,
        ),
    )
    policy, summary = trainer.run()
    return save_training_outputs(run_dir, policy, cfg, summary)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train DR-CIRL with frozen domain randomization.")
    parser.add_argument("--config", nargs="+")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(list(argv or sys.argv[1:]))
    overrides = {}
    if args.seed is not None:
        overrides["seed"] = int(args.seed)
    artifacts = train_dr_cirl(args.config if args.config else None, **overrides)
    print(json.dumps(artifacts, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
