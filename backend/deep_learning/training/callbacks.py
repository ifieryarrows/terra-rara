"""
Custom Lightning Callbacks for TFT-ASRO training.

CurriculumLossScheduler: Gradually shifts loss emphasis from calibration
to directional accuracy as training progresses.

StochasticWeightAveraging: Averages model weights over the last portion
of training to find flatter optima and improve generalisation.

References:
    - Bengio et al. (2009) "Curriculum Learning" (ICML)
    - Izmailov et al. (2018) "Averaging Weights Leads to Wider Optima" (UAI)
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    import lightning.pytorch as pl
except ImportError:
    import pytorch_lightning as pl  # type: ignore[no-redef]


class CurriculumLossScheduler(pl.Callback):
    """
    Gradually increase directional loss weight during training.

    Phase 1 (warmup_epochs): Model learns to calibrate — high quantile weight,
        low directional weight.  This establishes correct prediction scale
        before asking the model to learn direction.

    Phase 2 (remaining epochs): Directional components (Sharpe + MADL) are
        linearly ramped up to their target weights, forcing the model to
        learn direction on top of its calibration foundation.

    This prevents the model from being overwhelmed by conflicting gradients
    from calibration, direction, and volatility objectives simultaneously.
    """

    def __init__(
        self,
        warmup_epochs: int = 10,
        initial_lambda_quantile: float = 0.65,
        target_lambda_quantile: float = 0.35,
        initial_lambda_madl: float = 0.05,
        target_lambda_madl: float = 0.25,
    ):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.initial_lq = initial_lambda_quantile
        self.target_lq = target_lambda_quantile
        self.initial_madl = initial_lambda_madl
        self.target_madl = target_lambda_madl

    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        loss = pl_module.loss

        if not hasattr(loss, "lambda_quantile"):
            return

        if epoch < self.warmup_epochs:
            progress = epoch / max(self.warmup_epochs, 1)
            lq = self.initial_lq + (self.target_lq - self.initial_lq) * progress
            lm = self.initial_madl + (self.target_madl - self.initial_madl) * progress
        else:
            lq = self.target_lq
            lm = self.target_madl

        loss.lambda_quantile = lq
        if hasattr(loss, "lambda_madl"):
            loss.lambda_madl = lm

        if epoch % 10 == 0 or epoch == self.warmup_epochs:
            logger.info(
                "Curriculum epoch %d: lambda_quantile=%.3f (w_dir=%.3f) lambda_madl=%.3f",
                epoch, lq, 1.0 - lq, lm,
            )


class SWACallback(pl.Callback):
    """
    Stochastic Weight Averaging over the last ``swa_pct`` of training.

    Collects model weights from each epoch after the SWA start point
    and averages them at the end of training, producing a model that
    sits in a flatter region of the loss landscape with better
    generalisation properties.
    """

    def __init__(self, swa_start_pct: float = 0.75):
        super().__init__()
        self.swa_start_pct = swa_start_pct
        self._swa_state: dict | None = None
        self._n_averaged: int = 0

    def on_train_epoch_end(self, trainer, pl_module):
        max_epochs = trainer.max_epochs or 100
        swa_start = int(max_epochs * self.swa_start_pct)

        if trainer.current_epoch < swa_start:
            return

        state = pl_module.state_dict()
        if self._swa_state is None:
            import copy
            self._swa_state = copy.deepcopy(state)
            self._n_averaged = 1
        else:
            self._n_averaged += 1
            for key in self._swa_state:
                self._swa_state[key] = (
                    self._swa_state[key] * (self._n_averaged - 1) + state[key]
                ) / self._n_averaged

    def on_train_end(self, trainer, pl_module):
        if self._swa_state is not None and self._n_averaged > 1:
            pl_module.load_state_dict(self._swa_state)
            logger.info(
                "SWA: averaged %d checkpoints from epoch %d onwards",
                self._n_averaged,
                int((trainer.max_epochs or 100) * self.swa_start_pct),
            )
