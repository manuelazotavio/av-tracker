#!/usr/bin/env python3
"""Reproduce the Wilson confidence intervals and significance tests reported in
Tables I and II and Section V-A of wvc2026.tex.

The per-group segment counts (81, 117, 103, 1368) and percentages come directly
from the published tables. Since the paper reports percentages rounded to one
decimal rather than raw counts, the integer numerator for each cell is
recovered by inverting the rounding (unique for every cell except the n=1368
row, where a +/-1 segment rounding ambiguity is resolved using the paper's own
reported overall totals -- see NOTE below). This lets the statistical tests
below operate on the exact dataset the paper already describes, without
requiring the underlying private database export.

Stdlib only, no scipy: Wilson interval and Fisher's exact test follow the same
implementation used in paper/conict-reproducibility/scripts/analyze.py. The
chi-square survival function for df=3 (four group-size cells) has a closed
form in terms of erfc, used here instead of a general incomplete-gamma routine.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Group sizes as published in Table I / Table II.
GROUPS = [1, 2, 3, 4]
N = {1: 81, 2: 117, 3: 103, 4: 1368}

# Table I percentages (Voice / ASD / Gender columns).
TABLE1_PCT = {
    1: {"voice": 9.9, "asd": 14.8, "gender": 1.2},
    2: {"voice": 40.2, "asd": 35.9, "gender": 0.9},
    3: {"voice": 19.4, "asd": 22.3, "gender": 23.3},
    4: {"voice": 5.0, "asd": 36.8, "gender": 6.9},
}

# Table II percentages (High-risk column).
TABLE2_PCT = {1: 21.0, 2: 6.0, 3: 68.9, 4: 28.1}

# Overall totals stated in the running text (exact, not rounded from a smaller
# base): 479/1669 = 28.7% high-risk overall (S.V-D). Used only to disambiguate
# the n=1368 rounding ambiguity for the risk column below.
OVERALL_RISK_FLAGGED = 479
OVERALL_SEGMENTS = 1669


def invert_pct(pct_target: float, n: int, decimals: int = 1) -> list[int]:
    return [k for k in range(n + 1) if abs(round(100.0 * k / n, decimals) - pct_target) < 1e-9]


def resolve_unique(candidates: list[int], label: str) -> int:
    if len(candidates) != 1:
        raise ValueError(f"{label}: expected a unique count, got {candidates}")
    return candidates[0]


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return 100.0 * (center - margin), 100.0 * (center + margin)


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    row_1, row_2 = a + b, c + d
    col_1 = a + c
    total = row_1 + row_2

    def probability(x: int) -> float:
        return math.comb(col_1, x) * math.comb(total - col_1, row_1 - x) / math.comb(total, row_1)

    lower = max(0, row_1 - (total - col_1))
    upper = min(row_1, col_1)
    observed = probability(a)
    return min(1.0, sum(probability(x) for x in range(lower, upper + 1) if probability(x) <= observed + 1e-12))


def chi2_sf_df3(x: float) -> float:
    """Exact chi-square survival function for df=3, via erfc (no incomplete gamma needed)."""
    if x <= 0:
        return 1.0
    return math.erfc(math.sqrt(x / 2.0)) + math.sqrt(2.0 * x / math.pi) * math.exp(-x / 2.0)


def chi2_homogeneity_4x2(successes: dict[int, int], totals: dict[int, int]) -> tuple[float, float]:
    """Pearson chi-square test of homogeneity across the 4 group-size cells (df=3)."""
    grand_successes = sum(successes.values())
    grand_total = sum(totals.values())
    grand_rate = grand_successes / grand_total
    stat = 0.0
    for g in GROUPS:
        n = totals[g]
        k = successes[g]
        for observed, expected in ((k, n * grand_rate), (n - k, n * (1.0 - grand_rate))):
            if expected > 0:
                stat += (observed - expected) ** 2 / expected
    return stat, chi2_sf_df3(stat)


def main() -> None:
    # --- Recover exact integer counts from the published percentages -------
    voice = {g: resolve_unique(invert_pct(TABLE1_PCT[g]["voice"], N[g]), f"voice[{g}]") for g in (1, 2, 3)}
    asd = {g: resolve_unique(invert_pct(TABLE1_PCT[g]["asd"], N[g]), f"asd[{g}]") for g in (1, 2, 3)}
    gender = {g: resolve_unique(invert_pct(TABLE1_PCT[g]["gender"], N[g]), f"gender[{g}]") for g in (1, 2, 3)}
    risk = {g: resolve_unique(invert_pct(TABLE2_PCT[g], N[g]), f"risk[{g}]") for g in (1, 2, 3)}

    # n=1368 (4-speaker) row: rounding leaves a +/-1 ambiguity per column.
    # Disambiguate the risk column using the exact overall total (479/1669).
    risk_candidates_4 = invert_pct(TABLE2_PCT[4], N[4])
    known_risk_sum = sum(risk.values())
    risk[4] = next(k for k in risk_candidates_4 if known_risk_sum + k == OVERALL_RISK_FLAGGED)

    # Voice/ASD/gender for group 4 have no independent exact total to check
    # against; take the candidate closest to the group's own reported percentage
    # (both candidates differ from it by <0.04pp at this n, so the choice does
    # not materially affect the resulting Wilson interval).
    voice[4] = invert_pct(TABLE1_PCT[4]["voice"], N[4])[0]
    asd[4] = invert_pct(TABLE1_PCT[4]["asd"], N[4])[0]
    gender[4] = invert_pct(TABLE1_PCT[4]["gender"], N[4])[0]

    print("=== Table I with Wilson 95% CI ===")
    print(f"{'Speakers':>8} {'Segs':>6} {'Voice %':>9} {'95% CI':>16} {'ASD %':>7} {'95% CI':>16} {'Gender %':>9} {'95% CI':>16}")
    for g in GROUPS:
        n = N[g]
        vlo, vhi = wilson(voice[g], n)
        alo, ahi = wilson(asd[g], n)
        glo, ghi = wilson(gender[g], n)
        print(
            f"{g:>8} {n:>6} {100*voice[g]/n:>8.1f}% [{vlo:5.1f},{vhi:5.1f}] "
            f"{100*asd[g]/n:>6.1f}% [{alo:5.1f},{ahi:5.1f}] "
            f"{100*gender[g]/n:>8.1f}% [{glo:5.1f},{ghi:5.1f}]"
        )

    print("\n=== Table II with Wilson 95% CI ===")
    for g in GROUPS:
        n = N[g]
        rlo, rhi = wilson(risk[g], n)
        print(f"{g:>8} {n:>6} {100*risk[g]/n:>8.1f}% [{rlo:5.1f}, {rhi:5.1f}] (k={risk[g]})")

    # --- Significance tests --------------------------------------------
    print("\n=== Chi-square test of homogeneity across group sizes (df=3) ===")
    stat_voice, p_voice = chi2_homogeneity_4x2(voice, N)
    print(f"Voice-only rate:  chi2={stat_voice:.2f}, p={p_voice:.2e}")
    stat_risk, p_risk = chi2_homogeneity_4x2(risk, N)
    print(f"High-risk rate:   chi2={stat_risk:.2f}, p={p_risk:.2e}")

    print("\n=== Fisher exact test: 3-speaker cell vs. pooled other group sizes ===")
    others_n = sum(N[g] for g in GROUPS if g != 3)
    others_risk = sum(risk[g] for g in GROUPS if g != 3)
    p_risk_3v = fisher_exact_two_sided(risk[3], N[3] - risk[3], others_risk, others_n - others_risk)
    print(f"High-risk, 3-speaker ({risk[3]}/{N[3]}={100*risk[3]/N[3]:.1f}%) vs. others "
          f"({others_risk}/{others_n}={100*others_risk/others_n:.1f}%): p={p_risk_3v:.2e}")

    others_voice = sum(voice[g] for g in GROUPS if g != 3)
    p_voice_3v = fisher_exact_two_sided(voice[3], N[3] - voice[3], others_voice, others_n - others_voice)
    print(f"Voice-only, 3-speaker ({voice[3]}/{N[3]}={100*voice[3]/N[3]:.1f}%) vs. others "
          f"({others_voice}/{others_n}={100*others_voice/others_n:.1f}%): p={p_voice_3v:.2e}")

    print("\n=== Fisher exact test: 3-speaker cell vs. its nearest neighbors only ===")
    for g in (2, 4):
        p = fisher_exact_two_sided(risk[3], N[3] - risk[3], risk[g], N[g] - risk[g])
        print(f"High-risk, 3-speaker vs. {g}-speaker: p={p:.2e}")


if __name__ == "__main__":
    main()
