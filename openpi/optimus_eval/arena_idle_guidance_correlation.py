from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np


TASK_IDS = (1, 2, 3, 18, 19, 22, 25, 26)
FEATURES = (
    "idle_steps",
    "exact_zero_arm_action_steps",
    "stationary_while_state_moving_steps",
    "exact_zero_while_state_moving_steps",
    "idle_steps_per_segment",
    "stationary_while_state_moving_steps_per_segment",
    "exact_zero_while_state_moving_steps_per_segment",
    "idle_replan_windows_any",
    "idle_replan_windows_majority",
    "exact_zero_while_state_moving_replan_windows_any",
    "exact_zero_while_state_moving_replan_windows_majority",
    "longest_idle_run",
    "segment_count",
    "jerk_rms_relative",
    "length_mean",
)


def _default_runs(root: Path, legacy_root: Path) -> dict[str, Path]:
    logs = root / "logs"
    return {
        "16S_V0": logs / "robomemarena_extra8_pi05_finetuned_ckpt30000_memory_v0_self_original2048_fp32_batch8_20260730_163831/results.txt",
        "64S_V0": logs / "robomemarena_extra8_pi05_finetuned_ckpt30000_memory_v0_self_original2048_fp32_continuous_frame_v2_frames1_batch8_20260731_043343/results.txt",
        "64T_V0": logs / "robomemarena_extra8_pi05_finetuned_ckpt30000_memory_v0_self_original2048_fp32_continuous_frame_v2_frames3_batch8_20260730_164753/results.txt",
        "dense10S_V0": logs / "robomemarena_extra8_pi05_finetuned_ckpt30000_memory_v0_self_original2048_fp32_dense_frame_v3_frames1_batch8_20260731_070859/results.txt",
        "dense10T_V0": logs / "robomemarena_extra8_pi05_finetuned_ckpt30000_memory_v0_self_original2048_fp32_dense_frame_v3_frames3_batch8_20260731_042037/results.txt",
        "16S_V3re": logs / "robomemarena_extra8_pi05_finetuned_ckpt30000_memory_v3_re_self_original2048_fp32_nfe_floor3_batch8_20260730_195630/results.txt",
        "64S_V3re": logs / "robomemarena_extra8_pi05_finetuned_ckpt30000_memory_v3_re_self_original2048_fp32_nfe_floor3_continuous_frame_v2_frames1_batch8_20260731_090912/results.txt",
        "64T_V3re": logs / "robomemarena_extra8_pi05_finetuned_ckpt30000_memory_v3_re_self_original2048_fp32_nfe_floor3_continuous_frame_v2_frames3_batch8_20260730_213419/results.txt",
        "dense10T_V3re": logs / "robomemarena_extra8_pi05_finetuned_ckpt30000_memory_v3_re_self_original2048_fp32_nfe_floor3_dense_frame_v3_frames3_batch8_20260731_092143/results.txt",
        "dense10S_V1": logs / "robomemarena_extra8_pi05_finetuned_ckpt30000_memory_v1_self_original2048_fp32_dense_frame_v3_frames1_batch8_20260731_154219/results.txt",
        "dense10T_V1": logs / "robomemarena_extra8_pi05_finetuned_ckpt30000_memory_v1_self_original2048_fp32_dense_frame_v3_frames3_batch8_20260731_193721/results.txt",
        "Base": legacy_root / "logs/robomemarena_extra8_pi05_finetuned_ckpt30000_batch8_20260727_153642/results.txt",
    }


def parse_results(path: Path) -> dict[int, dict[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: dict[int, dict[str, float]] = {}
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if not row.get("task", "").isdigit():
                continue
            task = int(row["task"])
            rows[task] = {"tsr": float(row["TSR"]), "csr": float(row["CSR"])}
    missing = set(TASK_IDS) - set(rows)
    if missing:
        raise ValueError(f"{path} is missing tasks {sorted(missing)}")
    return rows


def load_quality(path: Path) -> dict[int, dict[str, float]]:
    rows: dict[int, dict[str, float]] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if row["dataset"] != "RoboMemArena-Extra8":
                continue
            task = int(row["task_id"])
            rows[task] = {feature: float(row[feature]) for feature in FEATURES}
    missing = set(TASK_IDS) - set(rows)
    if missing:
        raise ValueError(f"{path} is missing Arena tasks {sorted(missing)}")
    return rows


def _rank_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(x @ y / denom) if denom > 1e-12 else 0.0


def exact_spearman(x: np.ndarray, y: np.ndarray, permutations: np.ndarray) -> tuple[float, float]:
    x_rank = _rank_average(x)
    y_rank = _rank_average(y)
    observed = _corr(x_rank, y_rank)
    permuted = np.asarray([_corr(x_rank, y_rank[index]) for index in permutations])
    p_value = float((np.count_nonzero(np.abs(permuted) >= abs(observed) - 1e-12) + 1) / (len(permuted) + 1))
    return observed, p_value


def partial_correlation(x: np.ndarray, y: np.ndarray, controls: np.ndarray) -> float:
    design = np.column_stack([np.ones(len(x)), controls])
    x_residual = x - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    y_residual = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return _corr(x_residual, y_residual)


def _fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1.0) -> np.ndarray:
    design = np.column_stack([np.ones(len(x)), x])
    beta = np.zeros(2, dtype=np.float64)
    for _ in range(50):
        logits = np.clip(design @ beta, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (probability - y) + np.asarray([0.0, l2 * beta[1]])
        weights = probability * (1.0 - probability)
        hessian = design.T @ (weights[:, None] * design) + np.diag([1e-8, l2])
        update = np.linalg.solve(hessian, gradient)
        beta -= update
        if np.linalg.norm(update) < 1e-9:
            break
    return beta


def loocv_cross_entropy(feature: np.ndarray, delta: np.ndarray) -> tuple[float, float]:
    harmful = (delta < -1e-9).astype(np.float64)
    losses = []
    null_losses = []
    for held_out in range(len(feature)):
        train = np.arange(len(feature)) != held_out
        mean = float(feature[train].mean())
        scale = float(feature[train].std())
        scale = scale if scale > 1e-12 else 1.0
        x_train = (feature[train] - mean) / scale
        beta = _fit_logistic(x_train, harmful[train])
        x_test = (feature[held_out] - mean) / scale
        probability = float(1.0 / (1.0 + np.exp(-np.clip(beta[0] + beta[1] * x_test, -30.0, 30.0))))
        probability = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
        target = harmful[held_out]
        losses.append(-(target * math.log(probability) + (1.0 - target) * math.log(1.0 - probability)))
        null_probability = float((harmful[train].sum() + 0.5) / (train.sum() + 1.0))
        null_losses.append(
            -(target * math.log(null_probability) + (1.0 - target) * math.log(1.0 - null_probability))
        )
    return float(np.mean(losses)), float(np.mean(null_losses))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(output_dir: Path, panel: list[dict], correlations: list[dict]) -> None:
    import matplotlib.pyplot as plt

    run_names = list(dict.fromkeys(row["run"] for row in panel))
    matrix = np.asarray(
        [[next(row["delta_csr"] for row in panel if row["run"] == run and row["task_id"] == task) for task in TASK_IDS] for run in run_names]
    )
    figure, axis = plt.subplots(figsize=(12, 6))
    image = axis.imshow(matrix, cmap="RdYlGn", vmin=-max(40.0, abs(matrix).max()), vmax=max(40.0, abs(matrix).max()), aspect="auto")
    axis.set_xticks(range(len(TASK_IDS)), TASK_IDS)
    axis.set_yticks(range(len(run_names)), run_names)
    axis.set_xlabel("Arena task")
    axis.set_title("CSR delta relative to Pi30k Base")
    for row_index in range(len(run_names)):
        for column_index in range(len(TASK_IDS)):
            axis.text(column_index, row_index, f"{matrix[row_index, column_index]:.1f}", ha="center", va="center", fontsize=8)
    figure.colorbar(image, ax=axis, label="CSR delta (pp)")
    figure.tight_layout()
    figure.savefig(output_dir / "task_delta_heatmap.png", dpi=180)
    plt.close(figure)

    selected_runs = [name for name in ("dense10S_V0", "dense10T_V0", "dense10S_V1", "dense10T_V1") if name in run_names]
    figure, axes = plt.subplots(1, len(selected_runs), figsize=(4.3 * len(selected_runs), 4), sharey=True)
    if len(selected_runs) == 1:
        axes = [axes]
    for axis, run in zip(axes, selected_runs):
        rows = [row for row in panel if row["run"] == run]
        x = np.asarray([row["idle_steps"] for row in rows])
        y = np.asarray([row["delta_csr"] for row in rows])
        axis.scatter(x, y)
        for row in rows:
            axis.annotate(str(row["task_id"]), (row["idle_steps"], row["delta_csr"]), xytext=(3, 3), textcoords="offset points")
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(run)
        axis.set_xlabel("Strict-idle steps per trajectory")
    axes[0].set_ylabel("CSR delta vs Base (pp)")
    figure.tight_layout()
    figure.savefig(output_dir / "idle_vs_csr_delta.png", dpi=180)
    plt.close(figure)

    corr_runs = selected_runs
    corr_features = [
        "idle_steps",
        "exact_zero_arm_action_steps",
        "exact_zero_while_state_moving_steps",
        "exact_zero_while_state_moving_replan_windows_any",
    ]
    matrix = np.asarray(
        [[next(row["spearman"] for row in correlations if row["run"] == run and row["feature"] == feature and row["outcome"] == "delta_csr") for feature in corr_features] for run in corr_runs]
    )
    figure, axis = plt.subplots(figsize=(7, 4.5))
    image = axis.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    axis.set_xticks(range(len(corr_features)), corr_features, rotation=25, ha="right")
    axis.set_yticks(range(len(corr_runs)), corr_runs)
    axis.set_title("Task-level Spearman correlation with CSR delta")
    for row_index in range(len(corr_runs)):
        for column_index in range(len(corr_features)):
            axis.text(column_index, row_index, f"{matrix[row_index, column_index]:.2f}", ha="center", va="center")
    figure.colorbar(image, ax=axis, label="Spearman rho")
    figure.tight_layout()
    figure.savefig(output_dir / "idle_feature_correlation.png", dpi=180)
    plt.close(figure)


def analyze(root: Path, legacy_root: Path, quality_csv: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    quality = load_quality(quality_csv)
    run_paths = _default_runs(root, legacy_root)
    results = {name: parse_results(path) for name, path in run_paths.items()}
    base = results.pop("Base")
    panel = []
    for run, tasks in results.items():
        for task in TASK_IDS:
            row = {
                "run": run,
                "source": str(run_paths[run]),
                "task_id": task,
                "base_tsr": base[task]["tsr"],
                "base_csr": base[task]["csr"],
                "run_tsr": tasks[task]["tsr"],
                "run_csr": tasks[task]["csr"],
                "delta_tsr": tasks[task]["tsr"] - base[task]["tsr"],
                "delta_csr": tasks[task]["csr"] - base[task]["csr"],
                **quality[task],
            }
            panel.append(row)

    permutations = np.asarray(list(itertools.permutations(range(len(TASK_IDS)))), dtype=np.int16)
    correlations = []
    cross_entropy = []
    for run in results:
        rows = [row for row in panel if row["run"] == run]
        for outcome in ("delta_tsr", "delta_csr"):
            y = np.asarray([row[outcome] for row in rows], dtype=np.float64)
            base_outcome = np.asarray([row["base_" + outcome.removeprefix("delta_")] for row in rows])
            log_length = np.log(np.asarray([row["length_mean"] for row in rows]))
            for feature in FEATURES:
                x = np.asarray([row[feature] for row in rows], dtype=np.float64)
                spearman, exact_p = exact_spearman(x, y, permutations)
                controls = base_outcome[:, None] if feature == "length_mean" else np.column_stack([base_outcome, log_length])
                correlations.append(
                    {
                        "run": run,
                        "outcome": outcome,
                        "feature": feature,
                        "pearson": _corr(x, y),
                        "spearman": spearman,
                        "spearman_exact_p": exact_p,
                        "partial_pearson_control_base_length": partial_correlation(x, y, controls),
                        "tasks": len(TASK_IDS),
                    }
                )
                if outcome == "delta_csr":
                    model_loss, null_loss = loocv_cross_entropy(x, y)
                    cross_entropy.append(
                        {
                            "run": run,
                            "feature": feature,
                            "harmful_tasks": int(np.count_nonzero(y < -1e-9)),
                            "loocv_cross_entropy": model_loss,
                            "null_cross_entropy": null_loss,
                            "cross_entropy_gain": null_loss - model_loss,
                        }
                    )

    _write_csv(output_dir / "task_panel.csv", panel)
    _write_csv(output_dir / "correlations.csv", correlations)
    _write_csv(output_dir / "cross_entropy.csv", cross_entropy)
    provenance = {
        "quality_csv": str(quality_csv),
        "base": str(run_paths["Base"]),
        "runs": {name: str(path) for name, path in run_paths.items() if name != "Base"},
        "tasks": list(TASK_IDS),
        "notes": [
            "Correlations use eight task aggregates, not episode-level independent samples.",
            "Exact p-values enumerate all 8! task permutations.",
            "Cross entropy predicts whether task CSR delta is negative using leave-one-task-out logistic regression.",
        ],
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    _plot(output_dir, panel, correlations)
    return {"panel": panel, "correlations": correlations, "cross_entropy": cross_entropy, "provenance": provenance}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--legacy-root", type=Path, default=Path("/path/to/local/CVPR26-OptimusVLA/openpi"))
    parser.add_argument(
        "--quality-csv",
        type=Path,
        default=Path("/path/to/storage/datasets/robotics/RoboMemArena/derived/trajectory_quality_audit/episodes29_balanced/task_summary.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/path/to/storage/datasets/robotics/RoboMemArena/derived/trajectory_quality_audit/idle_guidance_task_correlation"),
    )
    args = parser.parse_args()
    result = analyze(args.root.resolve(), args.legacy_root.resolve(), args.quality_csv.resolve(), args.output_dir.resolve())
    key_rows = [
        row
        for row in result["correlations"]
        if row["outcome"] == "delta_csr" and row["feature"] == "idle_steps"
    ]
    print(json.dumps(key_rows, indent=2))
    print(f"Wrote idle/guidance task analysis to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
