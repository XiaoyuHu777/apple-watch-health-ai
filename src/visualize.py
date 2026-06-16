"""Visualization helpers for personal health trend analysis."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "apple-watch-health-ai-matplotlib"),
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    str(Path(tempfile.gettempdir()) / "apple-watch-health-ai-cache"),
)

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


DAILY_COLOR = "#8FB5CC"
TREND_COLOR = "#123F5D"
TEXT_COLOR = "#22313F"
MUTED_TEXT_COLOR = "#657786"
GRID_COLOR = "#DCE3E8"
GAP_COLOR = "#E8EDF1"

RECOVERY_BANDS = [
    (0, 30, "#FBEDEE", "LOW  0-30", "#A85A60"),
    (30, 60, "#FFF7E3", "MODERATE  30-60", "#9A772E"),
    (60, 100, "#EDF7F0", "STRONG  60-100", "#4D8060"),
]


def _prepare_recovery_series(
    frame: pd.DataFrame,
    rolling_window_days: int,
) -> pd.DataFrame:
    """Validate plot columns and calculate a display-only rolling mean."""
    required_columns = {"date", "recovery_score"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(
            "Recovery plot is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    plot_frame = frame[["date", "recovery_score"]].copy()
    plot_frame["date"] = pd.to_datetime(plot_frame["date"], errors="coerce")
    plot_frame["recovery_score"] = pd.to_numeric(
        plot_frame["recovery_score"],
        errors="coerce",
    )
    plot_frame = (
        plot_frame.dropna(subset=["date"])
        .drop_duplicates(subset=["date"], keep="last")
        .sort_values("date")
        .set_index("date")
    )
    if plot_frame.empty:
        raise ValueError("Recovery plot has no valid dates")

    rolling_score = plot_frame["recovery_score"].rolling(
        f"{rolling_window_days}D",
        min_periods=3,
    ).mean()
    # Only draw a rolling value where an observed score exists. This avoids
    # visually extrapolating the trend into periods without score coverage.
    plot_frame["rolling_mean"] = rolling_score.where(
        plot_frame["recovery_score"].notna()
    )
    return plot_frame


def _iter_contiguous_segments(
    series: pd.Series,
    max_gap_days: int,
) -> Iterable[pd.Series]:
    """Yield observed line segments without bridging long missing periods."""
    observed = series.dropna()
    if observed.empty:
        return

    segment_ids = (
        observed.index.to_series()
        .diff()
        .gt(pd.Timedelta(days=max_gap_days))
        .cumsum()
    )
    for _, segment in observed.groupby(segment_ids):
        yield segment


def _plot_segmented_line(
    axis: Axes,
    series: pd.Series,
    *,
    max_gap_days: int,
    label: str,
    color: str,
    linewidth: float,
    alpha: float,
    zorder: int,
) -> None:
    """Plot a time series while leaving long data gaps visually open."""
    for segment_index, segment in enumerate(
        _iter_contiguous_segments(series, max_gap_days)
    ):
        axis.plot(
            segment.index,
            segment.values,
            label=label if segment_index == 0 else None,
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=zorder,
        )


def _configure_date_axis(axis: Axes, date_span_days: int) -> None:
    """Choose readable date ticks based on the plotted time span."""
    if date_span_days <= 240:
        locator = mdates.MonthLocator(interval=1)
    elif date_span_days <= 730:
        locator = mdates.MonthLocator(interval=2)
    elif date_span_days <= 1_460:
        locator = mdates.MonthLocator(bymonth=[1, 4, 7, 10])
    else:
        locator = mdates.MonthLocator(interval=6)

    formatter = mdates.ConciseDateFormatter(locator)
    formatter.formats = ["%Y", "%b", "%b", "%d", "%H:%M", "%H:%M"]
    formatter.zero_formats = ["", "%Y", "%b\n%Y", "%d %b", "%H:%M", "%H:%M"]
    formatter.offset_formats = ["", "%Y", "%Y", "%Y-%b", "%Y-%b-%d", "%Y-%b-%d"]
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(formatter)


def _add_recovery_bands(
    axis: Axes,
    *,
    alpha: float,
    show_labels: bool,
) -> None:
    """Add subtle explanatory score bands and labels."""
    for lower, upper, fill_color, label, label_color in RECOVERY_BANDS:
        axis.axhspan(
            lower,
            upper,
            color=fill_color,
            alpha=alpha,
            linewidth=0,
            zorder=0,
        )
        if show_labels:
            label_x = 0.76 if upper == 100 else 0.992
            axis.text(
                label_x,
                (lower + upper) / 2,
                label,
                transform=axis.get_yaxis_transform(),
                ha="right",
                va="center",
                fontsize=8.2,
                fontweight="bold",
                color=label_color,
                alpha=0.78,
                zorder=1,
            )


def _find_data_gaps(
    series: pd.Series,
    max_gap_days: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Return gaps between observed scores that exceed the line threshold."""
    observed_dates = series.dropna().index
    gaps: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    for previous_date, next_date in zip(observed_dates[:-1], observed_dates[1:]):
        gap_days = (next_date - previous_date).days
        if gap_days > max_gap_days:
            gaps.append((previous_date, next_date, gap_days))
    return gaps


def _add_gap_indicators(
    axis: Axes,
    gaps: list[tuple[pd.Timestamp, pd.Timestamp, int]],
) -> None:
    """Add restrained vertical shading for long periods without scores."""
    for previous_date, next_date, _ in gaps:
        axis.axvspan(
            previous_date + pd.Timedelta(days=1),
            next_date - pd.Timedelta(days=1),
            color=GAP_COLOR,
            alpha=0.32,
            linewidth=0,
            zorder=0.5,
        )


def _annotate_extreme(
    axis: Axes,
    series: pd.Series,
    *,
    mode: str,
    label: str,
    xytext: tuple[int, int],
) -> None:
    """Annotate one rolling-mean extreme with a compact callout."""
    observed = series.dropna()
    if observed.empty:
        return

    date = observed.idxmax() if mode == "max" else observed.idxmin()
    value = observed.loc[date]
    axis.scatter(
        [date],
        [value],
        s=38,
        color=TREND_COLOR,
        edgecolor="white",
        linewidth=1.2,
        zorder=7,
    )
    axis.annotate(
        f"{label}  {value:.1f}\n{date:%b %Y}",
        xy=(date, value),
        xytext=xytext,
        textcoords="offset points",
        ha="left",
        va="bottom" if xytext[1] >= 0 else "top",
        fontsize=8.5,
        color=TEXT_COLOR,
        linespacing=1.25,
        arrowprops={
            "arrowstyle": "-",
            "color": MUTED_TEXT_COLOR,
            "linewidth": 0.8,
        },
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "white",
            "edgecolor": GRID_COLOR,
            "linewidth": 0.7,
            "alpha": 0.94,
        },
        zorder=8,
    )


def _recovery_level(score: float) -> str:
    """Return the descriptive band for a score."""
    if score < 30:
        return "low"
    if score < 60:
        return "moderate"
    return "strong"


def _add_summary_box(axis: Axes, plot_frame: pd.DataFrame) -> None:
    """Summarize current, typical, best, and lowest recovery observations."""
    latest_daily = plot_frame["recovery_score"].dropna()
    if latest_daily.empty:
        return

    latest_daily_date = latest_daily.index[-1]
    latest_daily_value = latest_daily.iloc[-1]
    latest_rolling = plot_frame["rolling_mean"].dropna()
    overall_mean = latest_daily.mean()

    axis.scatter(
        [latest_daily_date],
        [latest_daily_value],
        s=24,
        color=DAILY_COLOR,
        edgecolor="white",
        linewidth=0.8,
        zorder=7,
    )

    if latest_rolling.empty:
        rolling_summary = "Latest 14-day     N/A"
        best_summary = "Best 14-day       N/A"
        lowest_summary = "Lowest 14-day     N/A"
    else:
        latest_rolling_date = latest_rolling.index[-1]
        latest_rolling_value = latest_rolling.iloc[-1]
        best_date = latest_rolling.idxmax()
        best_value = latest_rolling.loc[best_date]
        lowest_date = latest_rolling.idxmin()
        lowest_value = latest_rolling.loc[lowest_date]
        axis.scatter(
            [latest_rolling_date],
            [latest_rolling_value],
            s=56,
            color=TREND_COLOR,
            edgecolor="white",
            linewidth=1.3,
            zorder=8,
        )
        rolling_summary = (
            f"Latest 14-day   {latest_rolling_value:5.1f}  "
            f"{latest_rolling_date:%d %b %Y}"
        )
        best_summary = (
            f"Best 14-day     {best_value:5.1f}  {best_date:%b %Y}"
        )
        lowest_summary = (
            f"Lowest 14-day   {lowest_value:5.1f}  {lowest_date:%b %Y}"
        )

    axis.text(
        0.985,
        0.955,
        (
            "RECOVERY SUMMARY\n"
            f"Latest score    {latest_daily_value:5.1f}  "
            f"{latest_daily_date:%d %b %Y}\n"
            f"{rolling_summary}\n"
            f"Overall mean    {overall_mean:5.1f}  "
            f"{_recovery_level(overall_mean).title()}\n"
            f"{best_summary}\n"
            f"{lowest_summary}"
        ),
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=8.6,
        fontfamily="monospace",
        color=TEXT_COLOR,
        linespacing=1.38,
        bbox={
            "boxstyle": "round,pad=0.6",
            "facecolor": "white",
            "edgecolor": GRID_COLOR,
            "linewidth": 0.8,
            "alpha": 0.96,
        },
        zorder=9,
    )


def plot_recovery_trend(
    frame: pd.DataFrame,
    *,
    version: str = "portfolio",
    rolling_window_days: int = 14,
    max_gap_days: int = 21,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """Create a clean or portfolio-ready personal recovery trend figure.

    This function changes only presentation. It reads the existing
    ``recovery_score`` values and calculates the same display-only rolling mean
    used by the previous chart.
    """
    if version not in {"clean", "portfolio"}:
        raise ValueError("version must be 'clean' or 'portfolio'")

    is_portfolio = version == "portfolio"
    if figsize is None:
        figsize = (14, 7) if is_portfolio else (13.5, 6.4)

    plot_frame = _prepare_recovery_series(frame, rolling_window_days)
    observed_scores = plot_frame["recovery_score"].dropna()
    if observed_scores.empty:
        raise ValueError("Recovery plot has no scored days")

    start_date = observed_scores.index.min()
    end_date = observed_scores.index.max()
    scored_days = len(observed_scores)
    date_span_days = max((end_date - start_date).days, 1)
    overall_mean = observed_scores.mean()
    gaps = _find_data_gaps(plot_frame["recovery_score"], max_gap_days)

    figure, axis = plt.subplots(figsize=figsize, facecolor="#FAFBFC")
    axis.set_facecolor("white")
    _add_recovery_bands(
        axis,
        alpha=0.58 if is_portfolio else 0.38,
        show_labels=True,
    )
    if is_portfolio:
        _add_gap_indicators(axis, gaps)

    _plot_segmented_line(
        axis,
        plot_frame["recovery_score"],
        max_gap_days=max_gap_days,
        label="Daily recovery score",
        color=DAILY_COLOR,
        linewidth=0.8,
        alpha=0.25 if is_portfolio else 0.20,
        zorder=2,
    )
    _plot_segmented_line(
        axis,
        plot_frame["rolling_mean"],
        max_gap_days=max_gap_days,
        label=f"{rolling_window_days}-day rolling mean",
        color=TREND_COLOR,
        linewidth=2.9 if is_portfolio else 2.5,
        alpha=0.98,
        zorder=5,
    )
    axis.axhline(
        overall_mean,
        color="#6F8797",
        linewidth=1.1,
        linestyle=(0, (4, 4)),
        alpha=0.82,
        label=f"Overall mean  {overall_mean:.1f}",
        zorder=3,
    )

    if is_portfolio:
        _annotate_extreme(
            axis,
            plot_frame["rolling_mean"],
            mode="max",
            label="Rolling peak",
            xytext=(12, 13),
        )
        _annotate_extreme(
            axis,
            plot_frame["rolling_mean"],
            mode="min",
            label="Rolling low",
            xytext=(12, -15),
        )
    _add_summary_box(axis, plot_frame)

    axis.set_ylim(0, 100)
    axis.set_yticks(range(0, 101, 20))
    axis.set_ylabel(
        "Recovery score",
        fontsize=11,
        color=TEXT_COLOR,
        labelpad=10,
    )
    axis.set_xlabel("")
    axis.set_xlim(
        start_date - pd.Timedelta(days=max(date_span_days * 0.015, 5)),
        end_date + pd.Timedelta(days=max(date_span_days * 0.025, 8)),
    )
    _configure_date_axis(axis, date_span_days)

    axis.set_title(
        "Personal Recovery Score Trend",
        loc="left",
        fontsize=20 if is_portfolio else 17,
        fontweight="bold",
        color=TEXT_COLOR,
        pad=42 if is_portfolio else 38,
    )
    subtitle = (
        f"{start_date:%b %Y} - {end_date:%b %Y}  |  "
        f"{scored_days:,} scored days  |  "
        f"Overall: {_recovery_level(overall_mean).title()} "
        f"({overall_mean:.1f})  |  "
        f"{rolling_window_days}-day window  |  "
        "Apple Watch / Apple Health daily metrics"
    )
    axis.text(
        0,
        1.035,
        subtitle,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color=MUTED_TEXT_COLOR,
    )

    axis.grid(
        axis="y",
        color=GRID_COLOR,
        linewidth=0.65,
        alpha=0.45 if is_portfolio else 0.32,
        zorder=1,
    )
    axis.grid(axis="x", visible=False)
    axis.tick_params(
        axis="both",
        labelsize=9.5,
        colors=MUTED_TEXT_COLOR,
        length=0,
        pad=7,
    )
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color(GRID_COLOR)
        axis.spines[spine].set_linewidth(0.8)

    legend = axis.legend(
        loc="upper left",
        bbox_to_anchor=(0, 1.002),
        ncol=3,
        frameon=False,
        fontsize=9.2,
        handlelength=2.6,
        columnspacing=1.5,
        borderaxespad=0,
    )
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    figure.text(
        0.075 if is_portfolio else 0.07,
        0.018,
        (
            "Shaded bands are descriptive guides, not clinical thresholds. "
            f"Lines break when observations are more than {max_gap_days} days apart."
            + (
                f" Muted vertical spans mark {len(gaps)} long data gaps."
                if is_portfolio and gaps
                else ""
            )
        ),
        ha="left",
        va="bottom",
        fontsize=8.5,
        color=MUTED_TEXT_COLOR,
    )
    figure.tight_layout(
        rect=(0.035, 0.055, 0.99, 0.95)
        if is_portfolio
        else (0.03, 0.055, 0.99, 0.95)
    )
    return figure


def save_figure_formats(
    figure: Figure,
    output_path: Path,
    *,
    dpi: int = 240,
) -> tuple[Path, Path]:
    """Save matching high-quality PNG and SVG files atomically."""
    png_path = output_path.with_suffix(".png")
    svg_path = output_path.with_suffix(".svg")
    png_path.parent.mkdir(parents=True, exist_ok=True)

    for target_path in (png_path, svg_path):
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=target_path.parent,
                prefix=f".{target_path.stem}.",
                suffix=target_path.suffix,
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)

            figure.savefig(
                temporary_path,
                dpi=dpi,
                bbox_inches="tight",
                facecolor=figure.get_facecolor(),
            )
            os.replace(temporary_path, target_path)
        except OSError:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise

    return png_path, svg_path
