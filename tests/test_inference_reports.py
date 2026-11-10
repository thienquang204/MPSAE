"""Fixed-checkpoint curves without reference overlays, plus cost units."""
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


def test_e1_omits_supplied_dense_reference(monkeypatch, tmp_path):
    captured = _capture(monkeypatch)
    rows = [{"backbone": "ff2048", "method": "csr", "k": 8,
             "top1_mean": 60, "top1_n": 1}]
    figures.figure_e1_curves(rows, tmp_path, "ff2048", measured_dense={"top1": 71.2})
    ax = captured[0].axes[0]
    assert [line.get_label() for line in ax.lines] == ["CSR"]
    assert list(ax.lines[0].get_ydata()) == [60]
    assert not ax.collections
    assert ax.get_xlabel() == "Train $K$"
    figures.figure_e1_curves(rows, tmp_path, "ff2048")
    assert len(captured[1].axes[0].lines) == 1  # no fallback to a published dense value


def test_all_imagenet_overviews_ignore_reference_inputs(monkeypatch, tmp_path):
    captured = _capture(monkeypatch)
    dense = {backbone: {"top1": 99, "probe_top1": 100}
             for backbone in ("ff2048", "vit_b16")}
    rows = [{"backbone": backbone, "method": "csr", "anchor_k": 8, "k": k,
             "top1_mean": 60 + index, "probe_top1_mean": 65 + index,
             "top1_n": 1, "probe_top1_n": 1, "nnz_mean": float(k)}
            for backbone in dense for index, k in enumerate((16, 32, 256))]
    figures._inference_figure(rows, tmp_path, "inference_summary", dense=dense)
    figures.figure_e2_generalisation(rows, tmp_path, "ff2048", measured_dense=dense["ff2048"])
    figures.figure_backbone_summary(rows, tmp_path, dense=dense)
    for fig in captured:
        for ax in fig.axes:
            # Each panel has only CSR plus the optional vertical training-K marker.
            curves = [line for line in ax.lines if line.get_label() == "CSR"]
            assert len(curves) == 1
            assert list(curves[0].get_xdata()) == [16, 32, 256]
            annotations = [line for line in ax.lines if line not in curves]
            for line in annotations:
                assert list(line.get_xdata()) == [8, 8]
            assert not ax.collections  # no reference points; inference K never equals 8
        assert {text.get_text() for legend in fig.legends for text in legend.get_texts()} == {"CSR"}


def test_retired_published_overlay_api_cannot_supply_reference_results():
    import mpsae.report as package
    from mpsae.report import reference

    assert "reference" not in package.__all__
    assert reference.PUBLISHED_SERIES == ()
    for backbone in ("ff2048", "rn18", "vit_b16"):
        assert reference.overlays(backbone) == []
        assert reference.dense_reference(backbone) is None
        for method in ("csr", "mrl", "csrv2_linear"):
            assert reference.published_curve(method, backbone) is None
            assert reference.published_label(method) is None
            assert reference.published_provenance(method) is None


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
    dense = record("dense", value=0.99)
    dense["metrics"].update(code_bytes_per_image=8192)
    raw = [dense]
    for anchor in (8, 64):
        for k in (2, 8, 64):
            item = record("csr", value=0.6 + k / 1000, train_k=anchor, eval_k=k)
            item["metrics"].update(code_bytes_per_image=k * 8, search_ms_per_query=k / 10)
            raw.append(item)
    rows = report.aggregate_records(raw)
    report._render_figures(rows, [], tmp_path)
    for fig in captured:
        for ax in fig.axes:
            # This also covers the storage plot, which formerly drew a dense marker.
            assert [line.get_label() for line in ax.lines] == ["CSR"]
            assert not ax.collections
        assert {text.get_text() for legend in fig.legends for text in legend.get_texts()} == {"CSR"}
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


def test_timing_axis_describes_top1_search_not_only_matrix_multiplication(monkeypatch, tmp_path):
    captured = _capture(monkeypatch)
    rows = [{"backbone": "ff2048", "method": "csr", "anchor_k": 8, "k": k,
             "retrieval_ms_mean": value, "retrieval_ms_n": 1}
            for k, value in ((16, 2.5), (32, 3.5))]
    figures.figure_e2_cost(rows, tmp_path, "ff2048")
    assert captured[0].axes[0].get_ylabel() == "Top-1 search (ms/batch)"


@pytest.mark.parametrize("group_count", [1, 2, 4, 5])
def test_inference_pages_keep_names_and_partition_training_groups(monkeypatch, tmp_path, group_count):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt

    groups = [("ff2048", anchor) for anchor in (8, 16, 32, 256)] + [("vit_b16", 8)]
    groups = groups[:group_count]
    rows = [{"backbone": backbone, "method": "csr", "anchor_k": anchor, "k": k,
             "top1_mean": group_index * 10 + index, "probe_top1_mean": group_index * 10 + index + 5,
             "top1_n": 1, "probe_top1_n": 1}
            for group_index, (backbone, anchor) in enumerate(groups)
            for index, k in enumerate((16, 32, 256))]
    exports = []

    def save(fig, out_dir, name, *, caption):
        exports.append((fig, name, caption))
        plt.close(fig)
        return [out_dir / f"{name}.pdf"]

    monkeypatch.setattr(figures, "save_figure", save)
    paths = figures._inference_figure(rows, tmp_path, "inference_summary")
    names = ["inference_summary"] + [f"inference_summary_part{page}"
                                    for page in range(2, (group_count + 1) // 2 + 1)]
    assert [name for _, name, _ in exports] == names
    assert [path.stem for path in paths] == names
    observed = []
    for page, (fig, _, caption) in enumerate(exports, start=1):
        assert len(fig.axes) <= 4  # at most two groups, with two metrics each
        assert fig.get_figheight() < 5.31
        if group_count > 2:
            assert caption.startswith(f"Part {page} of {len(names)}.")
        else:
            assert not caption.startswith("Part ")
        for probe_ax, top1_ax in zip(fig.axes[::2], fig.axes[1::2]):
            curve, = [line for line in top1_ax.lines if line.get_label() == "CSR"]
            group_index = int(curve.get_ydata()[0]) // 10
            assert list(curve.get_xdata()) == [16, 32, 256]
            assert list(curve.get_ydata()) == [group_index * 10 + i for i in range(3)]
            backbone, anchor = groups[group_index]
            title = f"{figures.backbone_label(backbone)} · Train $K$={anchor}"
            assert probe_ax.get_title() == top1_ax.get_title() == title
            observed.append((backbone, anchor))
    assert observed == groups
