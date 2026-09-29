# Web videos (gentle-care project page), 2026-09-29

Made by `scripts/make_web_videos_gentle.py` (`capture` / `probe` with .venv_myo, `compose` with .venv_mm).
Each episode is the unmodified `shoulder_task.run()`; frames and signals are recorded through read-only
hooks. The script checks that the harm it recomputes from the recorded signals equals `run()`'s harm.
All videos: H.264 (libx264, yuv420p), 30 fps, 1288 px wide, **0.25x speed** (shown on screen).
Harm = peak robot-caused joint-limit torque (baseline-subtracted, 50 ms moving average), shoulder + elbow,
as in `dev_r24_learned.harm`. The threshold is 0.5 N·m. Poster = last frame (final values).

| video | seed / person | strategies (offset m, angle deg, speed m/s) | measured (shown on screen) | stored grid value |
|---|---|---|---|---|
| lift_proximal_vs_distal.mp4 (11.3 s, 1.15 MB) | 104, unrestricted | proximal (-0.06, -40, 0.10) vs distal (+0.12, -40, 0.10) | proximal: shoulder +16.0°, elbow -2.5°; distal: shoulder +8.8°, elbow +9.6°; both done, harm 0 | - |
| fixed_vs_learned.mp4 (10.1 s, 1.15 MB) | 2258, elbow 7.2° free (shoulder unrestricted); record (90, 13.0) | F (0.03, -40, 0.10) vs L pick (-0.06, -20, 0.10) (tau 0.65, L score 0.76) | F peak 3.34 N·m (3.3409581826), done; L 0.00 N·m, done | round24/test.json: F 3.3409581826, L 0.0 (exact match) |
| learned_declines.mp4 (10.5 s, 0.61 MB) | 2248, shoulder 12.6°, elbow 8.7° free; record (15.9, 37.9) | F (0.03, -40, 0.10) vs L declines (best predicted safe rate 0.32 < 0.65) | F peak 4.30 N·m (4.2960916263), done | round24: 4.2960916263 (exact match) |
| force_guard_too_late.mp4 (9.9 s, 0.68 MB) | 2222, elbow 4.2° free (shoulder unrestricted) | F (0.03, -40, 0.10), unguarded | load first ≥ 0.5 N·m at 0.59 s (0.588); guard 25.0 N (25.05) trips at 0.64 s, when the peak load is already 1.35 N·m (1.3457); total peak 2.14 N·m (2.1390343323) | round24: 2.1390343323 (exact match) |

Guard threshold: round-21 guard for F = `round17/policy.json` guard[F] (50.10 N = 1.2 x p95 of harmless
wrist force) x 0.5 = 0.6 x p95 = 25.05 N. The trip is timed as in `run(stop_force=...)`: the end of the first
20 ms control step whose peak wrist force exceeds the threshold. Other harmful F persons probed
(`raw/probe_*.log`; seeds 2258, 2135, 2266, 2269, 2424, 2222, 2278, all elbow-restricted):
in all 7 the load crossed 0.5 N·m before the guard tripped, by 10 to 56 ms. This matches the "typical case" label and
round 21 (95 to 100% of harmful outcomes occur in the first attempt).

Wrist force = total probe contact force (`_instantaneous_harm()["force"]`, same signal as `peak_force`).
Raw frames, signals and run() dicts: `raw/*.npz`, `raw/*.json`; selection: `raw/selection.json`
(selection scripts `raw/select_*.py`); `raw/meta.json` = compose summary.
