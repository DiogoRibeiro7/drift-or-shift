"""Matplotlib helpers for visualizing experiment outcomes."""

from __future__ import annotations

import matplotlib.pyplot as plt
from numpy.typing import ArrayLike


def plot_risk_vs_prevalence(
    pi_tests: ArrayLike,
    risk_none: ArrayLike,
    risk_offset: ArrayLike,
    risk_oracle: ArrayLike,
    *,
    title: str | None = None,
):
    """Render risk versus prevalence curves for the three strategies."""
    fig, ax = plt.subplots()
    ax.plot(pi_tests, risk_none, label="No correction", marker="o")
    ax.plot(pi_tests, risk_offset, label="Offset correction", marker="s")
    ax.plot(pi_tests, risk_oracle, label="Oracle threshold", marker="^")
    ax.set_xscale("log")
    ax.set_xlabel("Test prevalence")
    ax.set_ylabel("Risk")
    ax.grid(True, which="both", linestyle=":")
    ax.legend()
    if title:
        ax.set_title(title)
    return fig


def plot_auc_pr_vs_prevalence(
    pi_tests: ArrayLike,
    auc_scores: ArrayLike,
    pr_auc_scores: ArrayLike,
    *,
    title: str | None = None,
):
    """Plot AUC and PR-AUC dependence on prevalence."""
    # Not sharex=True: sharing the axis while switching it to a log scale
    # makes matplotlib attempt a non-positive xlim and emit a warning on
    # every draw. Both panels plot the same prevalences, so their limits
    # come out identical anyway.
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax_auc, ax_pr = axes
    ax_auc.plot(pi_tests, auc_scores, marker="o")
    ax_auc.set_xscale("log")
    ax_auc.set_xlabel("Test prevalence")
    ax_auc.set_ylabel("ROC AUC")
    ax_auc.grid(True, linestyle=":")
    ax_pr.plot(pi_tests, pr_auc_scores, marker="o", color="tab:orange")
    ax_pr.set_xscale("log")
    ax_pr.set_xlabel("Test prevalence")
    ax_pr.set_ylabel("PR AUC")
    ax_pr.grid(True, linestyle=":")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_ess_vs_alpha(
    alpha: ArrayLike, ess_fraction: ArrayLike, *, title: str | None = None
):
    """Plot effective sample size fraction versus alpha."""
    fig, ax = plt.subplots()
    ax.plot(alpha, ess_fraction, marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("Alpha (class weight multiplier)")
    ax.set_ylabel("ESS fraction")
    ax.set_ylim(0, 1.05)
    ax.grid(True, which="both", linestyle=":")
    if title:
        ax.set_title(title)
    return fig
