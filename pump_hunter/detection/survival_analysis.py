"""
Pump survival analysis — models pump duration, retrace probability,
and optimal entry timing using historical pump event data.

Key outputs:
    - Time-to-peak distribution (Kaplan-Meier inspired)
    - Retrace probability conditioned on magnitude
    - Continuation probability (P of higher high)
    - Optimal entry delay model
"""

from __future__ import annotations

import datetime as dt
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog

from pump_hunter.config.settings import Settings
from pump_hunter.storage.database import Database

logger = structlog.get_logger(__name__)


class SurvivalAnalysis:
    """Analyzes pump event duration, retrace, and continuation statistics."""

    def __init__(self, settings: Settings, db: Database):
        self.cfg = settings.survival
        self.db = db

        # cached analysis results
        self._time_to_peak: Dict[str, float] = {}
        self._retrace_probs: Dict[float, float] = {}
        self._continuation_probs: Dict[float, float] = {}
        self._optimal_entry_delay: float = self.cfg.optimal_entry_delay_minutes
        self._last_update = 0.0
        self._event_count = 0

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._event_count >= self.cfg.min_events_for_analysis

    def get_survival_info(self, current_gain_pct: float = 0) -> dict:
        """Get survival statistics relevant to current pump state."""
        if not self.is_ready:
            return {"ready": False, "events": self._event_count}

        # find nearest retrace probability
        retrace = self._interpolate_retrace(current_gain_pct)
        continuation = self._interpolate_continuation(current_gain_pct)

        return {
            "ready": True,
            "events": self._event_count,
            "time_to_peak": self._time_to_peak,
            "retrace_probability": retrace,
            "continuation_probability": continuation,
            "optimal_entry_delay_min": round(self._optimal_entry_delay, 1),
        }

    def get_entry_adjustment(self) -> dict:
        """Return timing adjustment for entry."""
        return {
            "delay_minutes": self._optimal_entry_delay,
            "confidence": min(self._event_count / 100, 1.0),
        }

    # ------------------------------------------------------------------
    # update from DB
    # ------------------------------------------------------------------

    async def update(self) -> None:
        """Refresh survival statistics from recorded pump events."""
        now = time.time()
        interval = self.cfg.update_interval_hours * 3600
        if now - self._last_update < interval:
            return

        try:
            events = await self.db.get_completed_pumps_for_training()
            self._event_count = len(events)

            if self._event_count < self.cfg.min_events_for_analysis:
                logger.debug(
                    "survival_insufficient_data",
                    events=self._event_count,
                    needed=self.cfg.min_events_for_analysis,
                )
                return

            # extract pump metrics
            durations = []  # minutes to peak
            peak_gains = []  # max gain %
            retrace_data = []  # (peak_gain, retrace_pct)
            entry_delays = []  # optimal delay from signal to entry

            for event in events:
                features = event.features or {}
                duration = features.get("duration_minutes", 0)
                peak_gain = features.get("peak_gain_pct", 0)
                retrace = features.get("max_retrace_pct", 0)
                signal_to_move = features.get("signal_to_move_minutes", 0)

                if duration > 0:
                    durations.append(duration)
                if peak_gain > 0:
                    peak_gains.append(peak_gain)
                if peak_gain > 0 and retrace > 0:
                    retrace_data.append((peak_gain, retrace))
                if signal_to_move > 0:
                    entry_delays.append(signal_to_move)

            # compute distributions
            self._compute_time_to_peak(durations)
            self._compute_retrace_probabilities(retrace_data)
            self._compute_continuation(peak_gains)
            self._compute_optimal_entry(entry_delays)

            self._last_update = now

            logger.info(
                "survival_analysis_updated",
                events=self._event_count,
                median_duration=self._time_to_peak.get("median", 0),
                optimal_delay=round(self._optimal_entry_delay, 1),
            )

        except Exception as e:
            logger.error("survival_update_error", error=str(e))

    # ------------------------------------------------------------------
    # time-to-peak
    # ------------------------------------------------------------------

    def _compute_time_to_peak(self, durations: List[float]) -> None:
        """Build time-to-peak distribution."""
        if not durations:
            return

        arr = np.array(durations)
        bins = self.cfg.peak_time_bins_minutes

        self._time_to_peak = {
            "mean": round(float(np.mean(arr)), 1),
            "median": round(float(np.median(arr)), 1),
            "std": round(float(np.std(arr)), 1),
            "min": round(float(np.min(arr)), 1),
            "max": round(float(np.max(arr)), 1),
            "percentiles": {
                "p10": round(float(np.percentile(arr, 10)), 1),
                "p25": round(float(np.percentile(arr, 25)), 1),
                "p50": round(float(np.percentile(arr, 50)), 1),
                "p75": round(float(np.percentile(arr, 75)), 1),
                "p90": round(float(np.percentile(arr, 90)), 1),
            },
            "bin_probabilities": {},
        }

        # probability of peak occurring in each time bin
        for i, bin_max in enumerate(bins):
            bin_min = bins[i - 1] if i > 0 else 0
            count = np.sum((arr >= bin_min) & (arr < bin_max))
            prob = float(count / len(arr))
            self._time_to_peak["bin_probabilities"][f"{bin_min}-{bin_max}m"] = round(prob, 3)

        # probability of lasting beyond max bin
        beyond = np.sum(arr >= bins[-1])
        self._time_to_peak["bin_probabilities"][f">{bins[-1]}m"] = round(
            float(beyond / len(arr)), 3
        )

    # ------------------------------------------------------------------
    # retrace probability
    # ------------------------------------------------------------------

    def _compute_retrace_probabilities(
        self, retrace_data: List[Tuple[float, float]]
    ) -> None:
        """P(retrace ≥ X%) given peak magnitude."""
        if not retrace_data:
            return

        self._retrace_probs = {}

        for threshold in self.cfg.retrace_thresholds:
            # P(retrace ≥ threshold% of peak)
            retraces = [
                retrace / peak * 100
                for peak, retrace in retrace_data
                if peak > 0
            ]

            if retraces:
                arr = np.array(retraces)
                prob = float(np.mean(arr >= threshold))
                self._retrace_probs[threshold] = round(prob, 3)

                # also compute median time to retrace threshold
                # (would need richer data; approximate from available)

    # ------------------------------------------------------------------
    # continuation probability
    # ------------------------------------------------------------------

    def _compute_continuation(self, peak_gains: List[float]) -> None:
        """P(price makes new high) conditioned on current gain %."""
        if not peak_gains:
            return

        arr = np.array(peak_gains)
        self._continuation_probs = {}

        # for each gain level, P(gain > level)
        for level in [5, 10, 15, 20, 25, 30, 40, 50]:
            prob = float(np.mean(arr > level))
            self._continuation_probs[float(level)] = round(prob, 3)

    # ------------------------------------------------------------------
    # optimal entry delay
    # ------------------------------------------------------------------

    def _compute_optimal_entry(self, entry_delays: List[float]) -> None:
        """
        Find optimal entry delay — the time between signal firing
        and the best entry point (usually a small pullback).
        """
        if not entry_delays:
            return

        arr = np.array(entry_delays)

        # optimal delay = time that captures the most upside
        # use 25th percentile (early movers benefit)
        self._optimal_entry_delay = float(np.percentile(arr, 25))

    # ------------------------------------------------------------------
    # interpolation helpers
    # ------------------------------------------------------------------

    def _interpolate_retrace(self, current_gain: float) -> dict:
        """Interpolate retrace probability for current gain level."""
        result = {}
        for threshold, prob in self._retrace_probs.items():
            # adjust probability based on how far the pump has gone
            # further the pump, higher the retrace probability
            gain_factor = min(current_gain / 20, 2.0) if current_gain > 0 else 1.0
            adjusted = min(prob * gain_factor, 0.99)
            result[f"retrace_{threshold}pct"] = round(adjusted, 3)
        return result

    def _interpolate_continuation(self, current_gain: float) -> dict:
        """Interpolate continuation probability."""
        result = {}
        for level, prob in self._continuation_probs.items():
            if level > current_gain:
                result[f"reach_{level}pct"] = prob
        return result
