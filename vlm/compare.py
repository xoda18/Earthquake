"""
vlm/compare.py — Compare current crack analysis with previous run.

Programmatic JSON comparison: no LLM needed.  Computes deltas in crack
count, severity, dimensions, and flags new/resolved cracks.
"""

SEVERITY_ORDER = {"none": 0, "unknown": 0, "low": 1, "moderate": 2, "high": 3, "critical": 4}


def _worst_severity(reports: list[dict]) -> str:
    if not reports:
        return "none"
    return max(
        (r.get("severity", "none") for r in reports),
        key=lambda s: SEVERITY_ORDER.get(s, 0),
    )


def _total_cracks(reports: list[dict]) -> int:
    return sum(r.get("crack_count", 0) for r in reports)


def _max_length(reports: list[dict]) -> float:
    lengths = [r.get("max_crack_length_mm", 0) for r in reports]
    return max(lengths) if lengths else 0


def _max_width(reports: list[dict]) -> float:
    widths = [r.get("max_crack_width_mm", 0) for r in reports]
    return max(widths) if widths else 0


def _images_with_cracks(reports: list[dict]) -> set[str]:
    return {r["image_name"] for r in reports if r.get("has_crack")}


def compare_runs(current: list[dict], previous: list[dict]) -> dict:
    """Compare two sets of crack reports and return a structured diff.

    Parameters
    ----------
    current : list[dict]   — reports from this pipeline run
    previous : list[dict]  — reports from the most recent prior run

    Returns
    -------
    dict with keys:
        summary         — one-line human-readable summary
        crack_count     — {previous, current, delta}
        severity        — {previous, current, changed: bool}
        max_length_mm   — {previous, current, delta}
        max_width_mm    — {previous, current, delta}
        new_cracks      — list of image names with new cracks
        resolved_cracks — list of image names where cracks disappeared
        status          — "worsened" | "improved" | "stable" | "new"
    """
    if not previous:
        return {
            "summary": "First run — no previous data to compare.",
            "status": "new",
            "crack_count": {"previous": 0, "current": _total_cracks(current), "delta": _total_cracks(current)},
            "severity": {"previous": "none", "current": _worst_severity(current), "changed": True},
            "max_length_mm": {"previous": 0, "current": _max_length(current), "delta": _max_length(current)},
            "max_width_mm": {"previous": 0, "current": _max_width(current), "delta": _max_width(current)},
            "new_cracks": list(_images_with_cracks(current)),
            "resolved_cracks": [],
        }

    prev_count = _total_cracks(previous)
    curr_count = _total_cracks(current)
    prev_sev = _worst_severity(previous)
    curr_sev = _worst_severity(current)
    prev_len = _max_length(previous)
    curr_len = _max_length(current)
    prev_wid = _max_width(previous)
    curr_wid = _max_width(current)

    prev_cracked = _images_with_cracks(previous)
    curr_cracked = _images_with_cracks(current)
    new_cracks = list(curr_cracked - prev_cracked)
    resolved = list(prev_cracked - curr_cracked)

    # Determine overall status
    sev_prev = SEVERITY_ORDER.get(prev_sev, 0)
    sev_curr = SEVERITY_ORDER.get(curr_sev, 0)
    if sev_curr > sev_prev or curr_count > prev_count or curr_len > prev_len * 1.1:
        status = "worsened"
    elif sev_curr < sev_prev or curr_count < prev_count:
        status = "improved"
    else:
        status = "stable"

    # Build summary
    parts = []
    count_delta = curr_count - prev_count
    if count_delta > 0:
        parts.append(f"+{count_delta} new crack(s)")
    elif count_delta < 0:
        parts.append(f"{count_delta} fewer crack(s)")

    if curr_sev != prev_sev:
        parts.append(f"severity {prev_sev}→{curr_sev}")

    len_delta = curr_len - prev_len
    if abs(len_delta) > 0.5:
        parts.append(f"max length {len_delta:+.1f}mm ({prev_len:.1f}→{curr_len:.1f})")

    wid_delta = curr_wid - prev_wid
    if abs(wid_delta) > 0.05:
        parts.append(f"max width {wid_delta:+.2f}mm ({prev_wid:.2f}→{curr_wid:.2f})")

    if new_cracks:
        parts.append(f"new cracks in: {', '.join(new_cracks)}")
    if resolved:
        parts.append(f"resolved in: {', '.join(resolved)}")

    summary = "; ".join(parts) if parts else "No significant changes."

    return {
        "summary": summary,
        "status": status,
        "crack_count": {"previous": prev_count, "current": curr_count, "delta": count_delta},
        "severity": {"previous": prev_sev, "current": curr_sev, "changed": prev_sev != curr_sev},
        "max_length_mm": {"previous": prev_len, "current": curr_len, "delta": round(len_delta, 1)},
        "max_width_mm": {"previous": prev_wid, "current": curr_wid, "delta": round(wid_delta, 2)},
        "new_cracks": new_cracks,
        "resolved_cracks": resolved,
    }
