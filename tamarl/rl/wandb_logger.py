"""Weights & Biases logger for the DTA Markov Game training loop.

Provides a thin wrapper around wandb with graceful no-op when disabled.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class WandbLogger:
    """Thin wrapper around wandb for training metric logging.
    
    If enabled=False or wandb is unavailable, all methods are no-ops.
    """

    def __init__(
        self,
        project: str = "tamarl",
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = True,
        tags: Optional[List[str]] = None,
    ):
        self.enabled = enabled
        self._wandb = None
        self._run = None
        
        if not enabled:
            return
        
        try:
            import wandb
            self._wandb = wandb
            self._run = wandb.init(
                project=project,
                name=run_name,
                config=config or {},
                tags=tags,
                reinit=True,
            )
            
            # Use 'episode' as the custom x-axis for all metrics plotted
            self._wandb.define_metric("episode")
            self._wandb.define_metric("*", step_metric="episode")
            
            print(f"📊 W&B run: {self._run.url}")
        except ImportError:
            print("⚠️  wandb not installed, logging disabled")
            self.enabled = False
        except Exception as e:
            print(f"⚠️  wandb init failed: {e}, logging disabled")
            self.enabled = False

    def log_config(self, config: Dict[str, Any]):
        """Log additional config values after init."""
        if not self.enabled or self._wandb is None:
            return
        self._wandb.config.update(config, allow_val_change=True)

    def log_episode(self, episode: int, metrics: Dict[str, Any]):
        """Log per-episode metrics.
        
        Args:
            episode: episode index (0-based)
            metrics: dict of metric_name → value
        """
        if not self.enabled or self._wandb is None:
            return
        self._wandb.log({"episode": episode, **metrics})

    def log_summary(self, summary: Dict[str, Any]):
        """Log final summary metrics (appear in run summary table).
        
        Args:
            summary: dict of metric_name → value
        """
        if not self.enabled or self._wandb is None:
            return
        for key, val in summary.items():
            self._wandb.run.summary[key] = val



    def finish(self):
        """Finalize the wandb run."""
        if not self.enabled or self._wandb is None:
            return
        try:
            self._wandb.finish()
        except Exception:
            pass
