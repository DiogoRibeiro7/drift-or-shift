"""Check that the claims in RESULTS_DIGEST.md still reproduce.

A reproducibility repository is only as credible as the numbers it publishes.
The digest asserted specific risks and AUCs and nothing checked them, so a
refactor could quietly invalidate every documented result.

Two kinds of assertion here, deliberately weighted differently:

* The **orderings and invariances** are the paper's actual claims -- offset
  correction beats no correction under label shift, never beats the oracle,
  and fails under concept drift; ROC AUC is prevalence-invariant while PR-AUC
  is not. These are asserted strictly.
* The **exact figures** are asserted with a tolerance. They reproduce to four
  decimals locally, but LBFGS and BLAS differ subtly across platforms and CI
  runs on Linux, Windows, and macOS. The tolerance is wide enough to absorb
  that and narrow enough to catch a real regression.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from drift_or_shift.data_synth import resample_to_prevalence
from drift_or_shift.ess import ess_fraction
from drift_or_shift.models import fit_logistic_regression

# Tolerance for published figures. See the module docstring.
TOL = 5e-3


def _run(monkeypatch, tmp_path: Path, module_name: str, args: list[str]) -> dict:
    """Run an experiment CLI at a documented configuration, return its summary."""
    module = importlib.import_module(f"drift_or_shift.experiments.{module_name}")
    results_dir = tmp_path / "results"
    monkeypatch.setattr(
        sys, "argv", [module_name, *args, "--results-dir", str(results_dir)]
    )
    module.main()
    summary = next(iter(results_dir.rglob("*_summary.json")))
    return json.loads(summary.read_text(encoding="utf-8"))


def _by_prevalence(aggregated: dict, metric: str) -> dict[float, float]:
    return dict(zip(aggregated["pi_test"], aggregated[metric], strict=True))


# ---------------------------------------------------------------------------
# Experiment 1 -- synthetic label shift
# ---------------------------------------------------------------------------

EXP1_ARGS = [
    "--n-train",
    "2000",
    "--n-test",
    "2000",
    "--d",
    "6",
    "--pi-train",
    "0.2",
    "--seeds",
    "0",
    "1",
    "2",
    "3",
    "4",
]


@pytest.fixture(scope="module")
def exp1(tmp_path_factory, request):
    monkeypatch = pytest.MonkeyPatch()
    request.addfinalizer(monkeypatch.undo)
    payload = _run(
        monkeypatch,
        tmp_path_factory.mktemp("exp1"),
        "exp1_label_shift_synth",
        EXP1_ARGS,
    )
    return payload["aggregated"]


def test_exp1_offset_correction_never_loses_to_doing_nothing(exp1) -> None:
    """The central label-shift claim: the offset is free risk reduction."""
    none = _by_prevalence(exp1, "risk_none_mean")
    offset = _by_prevalence(exp1, "risk_offset_mean")

    for pi in none:
        assert offset[pi] <= none[pi] + 1e-9, f"offset lost at pi_test={pi}"


def test_exp1_offset_never_beats_the_oracle(exp1) -> None:
    """The oracle threshold is a lower bound; beating it means a broken metric."""
    offset = _by_prevalence(exp1, "risk_offset_mean")
    oracle = _by_prevalence(exp1, "risk_oracle_mean")

    for pi in offset:
        assert oracle[pi] <= offset[pi] + 1e-9, f"oracle lost at pi_test={pi}"


def test_exp1_gain_is_largest_under_the_strongest_shift(exp1) -> None:
    """pi_train is 0.2, so no correction is needed there and most is at 0.01."""
    none = _by_prevalence(exp1, "risk_none_mean")
    offset = _by_prevalence(exp1, "risk_offset_mean")

    gain = {pi: none[pi] - offset[pi] for pi in none}

    assert gain[0.01] > gain[0.1] > gain[0.2]
    assert gain[0.2] == pytest.approx(0.0, abs=1e-9), "corrected at the train prior"


@pytest.mark.parametrize(
    ("pi_test", "metric", "published"),
    [
        (0.01, "risk_none_mean", 0.0469),
        (0.01, "risk_offset_mean", 0.0084),
        (0.01, "risk_oracle_mean", 0.0078),
        (0.5, "risk_none_mean", 0.1699),
        (0.5, "risk_offset_mean", 0.1299),
        (0.5, "risk_oracle_mean", 0.1257),
    ],
)
def test_exp1_matches_the_published_figures(exp1, pi_test, metric, published) -> None:
    observed = _by_prevalence(exp1, metric)[pi_test]

    assert observed == pytest.approx(published, abs=TOL), (
        f"RESULTS_DIGEST.md publishes {metric}={published} at pi_test={pi_test}, "
        f"but this run produced {observed:.4f}"
    )


# ---------------------------------------------------------------------------
# Experiment 2 -- ROC AUC invariance, PR-AUC dependence
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def exp2(tmp_path_factory, request):
    monkeypatch = pytest.MonkeyPatch()
    request.addfinalizer(monkeypatch.undo)
    args = [
        "--n-train",
        "2000",
        "--n-test",
        "2000",
        "--d",
        "6",
        "--pi-train",
        "0.2",
        "--seeds",
        "0",
        "1",
        "2",
        "3",
        "4",
    ]
    payload = _run(
        monkeypatch, tmp_path_factory.mktemp("exp2"), "exp2_auc_pr_invariance", args
    )
    return payload["aggregated"]


def test_exp2_roc_auc_is_prevalence_invariant(exp2) -> None:
    """A constant logit offset preserves ranking, so ROC AUC must barely move."""
    auc = list(_by_prevalence(exp2, "auc_mean").values())

    assert max(auc) - min(auc) < 0.03, f"ROC AUC moved too much: {auc}"


def test_exp2_pr_auc_rises_with_prevalence(exp2) -> None:
    """PR-AUC depends on the positive rate; this is the contrast with ROC AUC."""
    pr = _by_prevalence(exp2, "pr_auc_mean")
    ordered = [pr[pi] for pi in sorted(pr)]

    assert ordered == sorted(ordered), f"PR-AUC not monotonic in prevalence: {pr}"
    assert pr[0.5] - pr[0.01] > 0.4, "PR-AUC barely moved; the contrast is the point"


def test_exp2_matches_the_published_figures(exp2) -> None:
    pr = _by_prevalence(exp2, "pr_auc_mean")

    assert pr[0.01] == pytest.approx(0.35, abs=0.02)
    assert pr[0.5] == pytest.approx(0.94, abs=0.02)


def test_exp2_roc_auc_stays_in_the_published_band(exp2) -> None:
    """The digest quotes [0.93, 0.96]; it previously quoted a band that the
    observed 0.9382 at pi_test=0.1 fell outside of."""
    auc = _by_prevalence(exp2, "auc_mean")

    for pi, value in auc.items():
        assert 0.93 <= value <= 0.96, f"ROC AUC {value:.4f} at pi_test={pi} is outside"


# ---------------------------------------------------------------------------
# Experiment 3 -- ESS against the class-weight multiplier
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("alpha", "published"), [(1, 1.00), (50, 0.23)])
def test_exp3_matches_the_published_ess_fractions(alpha, published) -> None:
    assert ess_fraction(1000, 0.2, alpha) == pytest.approx(published, abs=0.01)


def test_exp3_ess_decreases_monotonically_with_weight() -> None:
    """The paper's warning: aggressive weighting throws away effective data."""
    fractions = [ess_fraction(1000, 0.2, a) for a in (1, 5, 10, 20, 50)]

    assert fractions == sorted(fractions, reverse=True)


# ---------------------------------------------------------------------------
# Experiment 4 -- concept drift defeats the offset
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def exp4(tmp_path_factory, request):
    monkeypatch = pytest.MonkeyPatch()
    request.addfinalizer(monkeypatch.undo)
    args = [
        "--n-train",
        "2000",
        "--n-test",
        "2000",
        "--d",
        "6",
        "--pi-train",
        "0.2",
        "--seeds",
        "0",
        "1",
        "2",
    ]
    payload = _run(
        monkeypatch, tmp_path_factory.mktemp("exp4"), "exp4_concept_drift", args
    )
    return payload["aggregated"]


def test_exp4_offset_does_not_help_under_concept_drift(exp4) -> None:
    """The cautionary claim: when class-conditionals move, the offset is inert."""
    none = exp4["risk_none_mean"][0]
    offset = exp4["risk_offset_mean"][0]

    assert offset == pytest.approx(none, abs=1e-9), (
        "offset correction changed the risk under concept drift; "
        "the experiment's whole point is that it cannot"
    )


def test_exp4_retraining_recovers_what_the_offset_cannot(exp4) -> None:
    assert exp4["risk_retrain_mean"][0] < exp4["risk_none_mean"][0]


def test_exp4_matches_the_published_figures(exp4) -> None:
    assert exp4["risk_none_mean"][0] == pytest.approx(0.0995, abs=TOL)
    assert exp4["risk_offset_mean"][0] == pytest.approx(0.0995, abs=TOL)
    assert exp4["risk_retrain_mean"][0] == pytest.approx(0.0968, abs=TOL)


# ---------------------------------------------------------------------------
# Experiment 5 -- breast cancer, which is NOT reproducible across platforms
# ---------------------------------------------------------------------------

EXP5_UNSTABLE = (
    "exp5 fits logistic regression on raw, unscaled breast-cancer features "
    "(feature means span 0.004 to 880), so LBFGS exhausts max_iter without "
    "converging and the coefficients depend on the platform's BLAS. Its "
    "published figures reproduce exactly on Windows and differ on Linux and "
    "macOS -- at pi_test=0.5 the sign of the offset's effect even flips. "
    "Pinning those figures is only meaningful once the fit converges; see "
    "test_exp5_does_not_converge_on_raw_features."
)


@pytest.fixture(scope="module")
def exp5(tmp_path_factory, request):
    monkeypatch = pytest.MonkeyPatch()
    request.addfinalizer(monkeypatch.undo)
    args = ["--pi-train", "0.2", "--seeds", "0", "1", "2", "3", "4"]
    payload = _run(
        monkeypatch,
        tmp_path_factory.mktemp("exp5"),
        "exp5_realdata_breast_cancer",
        args,
    )
    return payload["aggregated"]


def test_exp5_does_not_converge_on_raw_features() -> None:
    """Pin the cause of exp5's irreproducibility so it stays visible.

    This is the defect behind every skipped assertion below. Standardizing the
    features makes the same fit converge in roughly twenty iterations, which
    would make the experiment reproducible -- but it would also change exp5's
    published numbers, so it is a call for the maintainers rather than a
    silent re-baseline.
    """
    data = load_breast_cancer()
    X_train, _, y_train, _ = train_test_split(
        data.data, data.target, test_size=0.5, stratify=data.target, random_state=0
    )
    rng = np.random.default_rng(1)
    X, y = resample_to_prevalence(X_train, y_train, 0.2, rng)

    raw = fit_logistic_regression(X, y, rng=np.random.default_rng(1))
    scaled = fit_logistic_regression(
        StandardScaler().fit_transform(X), y, rng=np.random.default_rng(1)
    )

    assert int(raw.n_iter_[0]) >= raw.max_iter, (
        "exp5's fit now converges on raw features; if that is intentional, "
        "re-derive RESULTS_DIGEST.md and re-enable the skipped tests below"
    )
    assert int(scaled.n_iter_[0]) < 100, "standardizing should converge quickly"


def test_exp5_oracle_remains_a_lower_bound(exp5) -> None:
    """Structural, so it holds regardless of where the optimizer stopped."""
    oracle = _by_prevalence(exp5, "risk_oracle_mean")
    none = _by_prevalence(exp5, "risk_none_mean")

    for pi in oracle:
        assert oracle[pi] <= none[pi] + 1e-9, f"oracle lost at pi_test={pi}"


def test_exp5_offset_helps_under_the_strongest_shift(exp5) -> None:
    """At pi_test=0.01 the offset is about -3.2 logits, which dominates the
    coefficient noise that makes the mid-grid results platform-dependent."""
    none = _by_prevalence(exp5, "risk_none_mean")
    offset = _by_prevalence(exp5, "risk_offset_mean")

    assert offset[0.01] < none[0.01]


@pytest.mark.skip(reason=EXP5_UNSTABLE)
def test_exp5_matches_the_published_figures(exp5) -> None:
    none = _by_prevalence(exp5, "risk_none_mean")
    offset = _by_prevalence(exp5, "risk_offset_mean")
    oracle = _by_prevalence(exp5, "risk_oracle_mean")

    assert none[0.01] == pytest.approx(0.0246, abs=TOL)
    assert offset[0.01] == pytest.approx(0.0091, abs=TOL)
    assert oracle[0.01] == pytest.approx(0.0035, abs=TOL)
    assert none[0.5] == pytest.approx(0.0484, abs=TOL)
    assert offset[0.5] == pytest.approx(0.0519, abs=TOL)
    assert oracle[0.5] == pytest.approx(0.0421, abs=TOL)


@pytest.mark.skip(reason=EXP5_UNSTABLE)
def test_exp5_offset_is_worse_at_the_balanced_prevalence(exp5) -> None:
    """Reproduces on Windows, but not on Linux, where the offset wins here."""
    none = _by_prevalence(exp5, "risk_none_mean")
    offset = _by_prevalence(exp5, "risk_offset_mean")

    assert offset[0.5] > none[0.5]
