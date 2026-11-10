"""Fixed-checkpoint curves, measurement-only references and cost units."""
import pytest

from mpsae.report import figures
from mpsae.benchmark import report


def _capture(monkeypatch):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    captured = []

    def save(fig, *args, **kwargs):
        fig.canvas.draw()
        captured.append(fig)
        plt.close(fig)
        return []

    monkeypatch.setattr(figures, "save_figure", save)
    monkeypatch.setattr("mpsae.report.style.save_figure", save)
    return captured


def test_e1_uses_only_experiment_measurements(monkeypatch, tmp_path):
    captured = _capture(monkeypatch)
    rows = [{"backbone": "ff2048", "method": "csr", "k": 8,
             "top1_mean": 60, "top1_n": 1}]
    figures.figure_e1_curves(rows, tmp_path, "ff2048", measured_dense={"top1": 71.2})
    ax = captured[0].axes[0]
    assert [line.get_label() for line in ax.lines] == ["CSR", "Dense"]
    assert list(ax.lines[-1].get_ydata()) == [71.2, 71.2]
    assert ax.get_xlabel() == "Train $K$"
    figures.figure_e1_curves(rows, tmp_path, "ff2048")
    assert len(captured[1].axes[0].lines) == 1  # no fallback to a published dense value


def test_inference_accuracy_and_nonzeros_keep_each_anchor_separate(monkeypatch, tmp_path):
    captured = _capture(monkeypatch)
    rows = [{"backbone": "ff2048", "method": "csr", "anchor_k": anchor, "k": k,
             "top1_mean": anchor + i, "probe_top1_mean": anchor + i + 2,
             "nnz_mean": nnz, "top1_n": 1, "probe_top1_n": 1}
            for anchor in (8, 64) for i, (k, nnz) in enumerate(((2, 2), (8, 7), (64, 6)))]
    figures.figure_e2_generalisation(rows, tmp_path, "ff2048")
    assert len(captured[0].axes) == 4
    for ax, anchor, probe in zip(captured[0].axes, (8, 8, 64, 64), (True, False, True, False)):
        curve, = [line for line in ax.lines if line.get_label() == "CSR"]
        assert list(curve.get_xdata()) == [2, 8, 64]
        assert list(curve.get_ydata()) == [anchor + i + (2 if probe else 0) for i in range(3)]
        assert ax.get_ylabel() == ("Linear probe (%)" if probe else "1-NN (%)")
    figures.figure_e2_nnz(rows, tmp_path, "ff2048")
    for ax in captured[1].axes:
        curve, = [line for line in ax.lines if line.get_label() == "CSR"]
        assert list(curve.get_xdata()) == [2, 7, 6]  # order by inference K, not measured nnz


def test_downstream_sweep_rejects_different_checkpoints():
    from test_benchmark_report import record
    first, second = record(eval_k=2), record(eval_k=64)
    second["checkpoint_id"] = "different-weights"
    with pytest.raises(ValueError, match="different checkpoints"):
        report.aggregate_records([first, second])


def test_legacy_retrieval_wall_time_is_converted_per_query_before_aggregation():
    from test_benchmark_report import record
    first, second = record(seed=1), record(seed=2)
    first["metrics"].update(seconds=2.0, queries=100)
    second["metrics"].update(seconds=3.0, queries=300)
    for item in (first, second):
        item["metrics"].pop("query_count")
    row, = report.aggregate_records([first, second], expected_seeds=(1, 2))
    assert row["search_ms_per_query_mean"] == 15  # mean of 20 and 10 ms/query
    assert row["search_ms_per_query_n"] == 2


def test_downstream_quality_and_cost_units_and_fixed_training_budget(monkeypatch, tmp_path):
    from test_benchmark_report import record
    captured = _capture(monkeypatch)
    raw = []
    for anchor in (8, 64):
        for k in (2, 8, 64):
            item = record("csr", value=0.6 + k / 1000, train_k=anchor, eval_k=k)
            item["metrics"].update(code_bytes_per_image=k * 8, search_ms_per_query=k / 10)
            raw.append(item)
    rows = report.aggregate_records(raw)
    report._render_figures(rows, [], tmp_path)
    quality = [fig for fig in captured if fig.axes[0].get_xlabel() == "Inference $K$"
               and fig.axes[0].get_ylabel() == "R@1 (%)"]
    assert len(quality) == 2
    for fig in quality:
        line, = fig.axes[0].lines
        assert list(line.get_xdata()) == [2, 8, 64]
        assert list(line.get_ydata()) == pytest.approx([60.2, 60.8, 66.4])
        assert line.get_label() == "CSR"
    costs = [fig for fig in captured if fig.axes[0].get_ylabel() == "Search (ms/query)"]
    assert len(costs) == 2
    for fig in costs:
        assert list(fig.axes[0].lines[0].get_ydata()) == [0.2, 0.8, 6.4]
        assert list(fig.axes[1].lines[0].get_ydata()) == [16 / 1024, 64 / 1024, 512 / 1024]
