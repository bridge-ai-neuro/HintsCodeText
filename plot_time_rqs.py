import pandas as pd
import numpy as np
from scipy.stats import sem, ranksums, ks_2samp
from itertools import product
from typing import List, Optional
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from utils import *
import os


def get_avg_time_with_hints_stats(
    df_to_plot: pd.DataFrame,
    y_label: str,
):
    combined_hints = {"Any Hint": HINT_ORDER[1:]}

    global PART_LABELS
    PART_LABELS = ["Clear"]

    stats = {}
    for hint_type, stim_pres, part_lbl in product(
        ["No Hint", "Any Hint"], STIMULUS_ORDER, PART_ORDER
    ):
        key = (hint_type, stim_pres, part_lbl)

        if hint_type == "No Hint":
            scores = df_to_plot[
                (df_to_plot["hint_type"] == hint_type)
                & (df_to_plot["stimulus_presentation"] == stim_pres)
                & (df_to_plot["part_lbl"] == part_lbl)
            ][y_label]
        else:
            scores = df_to_plot[
                (df_to_plot["hint_type"].isin(combined_hints[hint_type]))
                & (df_to_plot["stimulus_presentation"] == stim_pres)
                & (df_to_plot["part_lbl"] == part_lbl)
            ][y_label]

        stats[key] = {"mean": scores.mean(), "sem": sem(scores), "scores": scores}

    return stats


def plot_compare_avg_time_with_hints(
    df_to_plot: pd.DataFrame,
    y_label: str,
    y_title: str,
    ax: Axes,
    cb_palette: Optional[List] = None,
    chance_perf: Optional[float] = None,
    ylim: Optional[float] = None,
):
    stats = get_avg_time_with_hints_stats(df_to_plot, y_label)

    bar_width = 0.3
    r1 = np.arange(1)
    r2 = [x + bar_width for x in r1]
    p_val_pos = 1.50

    for i, stimuli in enumerate(STIMULUS_ORDER):
        for j, part_lbl in enumerate(PART_LABELS):

            print(f"Stimuli: {stimuli}, Part: {part_lbl}")

            none_key = ("No Hint", stimuli, part_lbl)
            any_exp_key = ("Any Hint", stimuli, part_lbl)

            none_value = stats[none_key]["mean"]
            none_sem = stats[none_key]["sem"]
            any_exp_value = stats[any_exp_key]["mean"]
            any_exp_sem = stats[any_exp_key]["sem"]

            _, p_value_ks = ks_2samp(
                stats[none_key]["scores"],
                stats[any_exp_key]["scores"],
                alternative="greater",
            )
            if p_value_ks > 0.05:
                p_value = ranksums(
                    stats[none_key]["scores"],
                    stats[any_exp_key]["scores"],
                    alternative="greater",
                ).pvalue
            else:
                p_value = 999

            print(f"\tp-value (none great) for {stimuli} and {part_lbl}: {p_value}")
            print(
                f'\tEffect Size: {cohen_d(stats[none_key]["scores"],stats[any_exp_key]["scores"])}'
            )
            print(
                f"\tDifference of means: {none_value-any_exp_value} [Means: {none_value} vs {any_exp_value}]"
            )

            if any_exp_value - none_value >= 0:
                ax.bar(
                    r1[j] + i * bar_width,
                    none_value,
                    width=bar_width,
                    color=cb_palette[i],
                    label=f"{stimuli}" if j == 0 else "",
                    yerr=none_sem,
                    edgecolor="black",
                )

                ax.bar(
                    r1[j] + i * bar_width,
                    any_exp_value - none_value,
                    width=bar_width,
                    bottom=none_value,
                    color=cb_palette[i],
                    label=f"{stimuli} + Hint" if j == 0 else "",
                    hatch="//",
                    alpha=0.5,
                    yerr=any_exp_sem,
                    edgecolor="black",
                )
            else:
                ax.bar(
                    r1[j] + i * bar_width,
                    none_value - any_exp_value,
                    bottom=any_exp_value,
                    width=bar_width,
                    color=cb_palette[i],
                    label=f"{stimuli}" if j == 0 else "",
                    yerr=none_sem,
                    edgecolor="black",
                )

                ax.bar(
                    r1[j] + i * bar_width,
                    any_exp_value,
                    width=bar_width,
                    color=cb_palette[i],
                    label=f"{stimuli} + Hint" if j == 0 else "",
                    hatch="//",
                    alpha=0.5,
                    yerr=any_exp_sem,
                    edgecolor="black",
                )

            # Add significance marker for No Hint vs Any Hint
            if p_value < 0.05:
                stars = get_asterisks(p_value)
                ax.text(
                    r1[j] + i * bar_width,
                    p_val_pos,
                    stars,
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

    # Add significance testing between Python and Text for No Hint
    python_no_hint = stats[("No Hint", "Python", "Clear")]["scores"]
    text_no_hint = stats[("No Hint", "Text", "Clear")]["scores"]

    p_value_ks = ks_2samp(text_no_hint, python_no_hint, alternative="greater").pvalue
    if p_value_ks > 0.05:
        p_value_stimulus = ranksums(
            text_no_hint, python_no_hint, alternative="greater"
        ).pvalue
        print(f"P-value (Python < Text) for No Hint: {p_value_stimulus}")
        print(f"\tEffect Size: {cohen_d(text_no_hint,python_no_hint)}")
        print(f"\tDifference of means: {mean(text_no_hint)-mean(python_no_hint)}")
    else:
        p_value_stimulus = 999

    if p_value_stimulus < 0.05:
        buffer = 0.38
        stars = get_asterisks(p_value_stimulus)
        bar_centers = [r1[0], r1[0] + bar_width]
        bar_heights = [
            stats[("No Hint", "Python", "Clear")]["mean"] + buffer,
            stats[("No Hint", "Text", "Clear")]["mean"] + buffer,
        ]
        max_height = max(bar_heights) + max(
            stats[("No Hint", "Python", "Clear")]["sem"],
            stats[("No Hint", "Text", "Clear")]["sem"],
        )

        # Draw the bracket
        bracket_height = max_height + 0.1 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        ax.plot(
            [bar_centers[0], bar_centers[0], bar_centers[1], bar_centers[1]],
            [max_height, bracket_height, bracket_height, max_height],
            color="black",
            linewidth=1.5,
        )

        # Add asterisks
        ax.text(
            np.mean(bar_centers),
            bracket_height,
            stars,
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_xticks([r + bar_width / 2 for r in r1])
    ax.set_xticklabels(PART_LABELS)
    ax.set_title("(a)")

    ax.set_ylabel(y_title, labelpad=15)

    if ylim:
        ax.set_ylim(0, ylim)

    if chance_perf:
        ax.axhline(y=chance_perf, color="r", linestyle="--")

    ax.legend()


def plot_compare_avg_time_with_hints_across_stimuli(
    df_to_plot: pd.DataFrame,
    y_label: str,
    y_title: str,
    axes: List[Axes],
    cb_palette: Optional[List] = None,
    chance_perf: Optional[float] = None,
    ylim: Optional[float] = None,
):
    titles = ["(b)", "(c)"]
    for i, (stimuli, part_lbl) in enumerate(product(STIMULUS_ORDER, PART_LABELS)):
        ax = axes[i]

        filtered_data = df_to_plot[
            (df_to_plot["stimulus_presentation"] == stimuli)
            & (df_to_plot["part_lbl"] == part_lbl)
        ]

        stats = {
            hint_type: {
                "scores": filtered_data[filtered_data["hint_type"] == hint_type][
                    y_label
                ]
            }
            for hint_type in HINT_ORDER
        }

        print(f"Stimuli: {stimuli}, Part: {part_lbl}")

        for hint_type in stats:
            stats[hint_type].update(
                {
                    "mean": stats[hint_type]["scores"].mean(),
                    "sem": sem(stats[hint_type]["scores"]),
                }
            )

        none_scores = stats["No Hint"]["scores"]
        p_values = {}
        for hint_type in stats:
            print(f"\tHint Type: {hint_type}")
            if hint_type != "No Hint":

                _, p_value_ks = ks_2samp(
                    none_scores, stats[hint_type]["scores"], alternative="greater"
                )
                if p_value_ks > 0.05:
                    p_val = ranksums(
                        none_scores,
                        stats[hint_type]["scores"],
                        alternative="greater",
                    ).pvalue
                    p_values[hint_type] = p_val
                else:
                    p_values[hint_type] = 999

                print(f"\t\tWilcoxon rank-sum test: {p_values[hint_type]}")
                print(
                    f'\t\tEffect Size: {cohen_d(none_scores,stats[hint_type]["scores"])}'
                )
                print(
                    f'\t\tDifference of means: {stats["No Hint"]["mean"]-stats[hint_type]["mean"]}'
                )

        plot_data = pd.DataFrame(
            {
                "hint_type": [hint_type for hint_type in HINT_ORDER],
                "mean": [stats[exp_type]["mean"] for exp_type in HINT_ORDER],
                "sem": [stats[exp_type]["sem"] for exp_type in HINT_ORDER],
                "p-values": [1]
                + [p_values.get(exp_type, 1) for exp_type in HINT_ORDER[1:]],
            }
        )

        no_hint_data = plot_data[plot_data["hint_type"] == "No Hint"].iloc[0]
        hint_data = plot_data[plot_data["hint_type"] != "No Hint"]

        for j, (_, row_hint) in enumerate(hint_data.iterrows()):
            if row_hint["mean"] - no_hint_data["mean"] >= 0:
                ax.bar(
                    j,
                    no_hint_data["mean"],
                    color=cb_palette[0] if stimuli == "Python" else cb_palette[1],
                    edgecolor="black",
                )

                ax.bar(
                    j,
                    row_hint["mean"] - no_hint_data["mean"],
                    bottom=no_hint_data["mean"],
                    color=cb_palette[0] if stimuli == "Python" else cb_palette[1],
                    alpha=0.5,
                    hatch="//",
                    edgecolor="black",
                )

            else:
                ax.bar(
                    j,
                    row_hint["mean"],
                    color=cb_palette[0] if stimuli == "Python" else cb_palette[1],
                    alpha=0.5,
                    hatch="//",
                    edgecolor="black",
                )

                ax.bar(
                    j,
                    no_hint_data["mean"] - row_hint["mean"],
                    bottom=row_hint["mean"],
                    color=cb_palette[0] if stimuli == "Python" else cb_palette[1],
                    edgecolor="black",
                )

            ax.errorbar(
                x=j,
                y=row_hint["mean"],
                yerr=row_hint["sem"],
                fmt="none",
                color="black",
            )
            ax.errorbar(
                x=j,
                y=no_hint_data["mean"],
                yerr=no_hint_data["sem"],
                fmt="none",
                color="black",
            )

            if row_hint["p-values"] < 0.05:
                stars = get_asterisks(row_hint["p-values"])
                ax.text(
                    j,
                    1.50,
                    stars,
                    ha="center",
                    va="bottom",
                    color="black",
                )

        ax.set_xticks(range(len(hint_data)))
        ax.set_xticklabels(hint_data["hint_type"])
        ax.set_title(f"{titles[i]}")

        if chance_perf is not None:
            ax.axhline(y=chance_perf, color="r", linestyle="--")

    if ylim:
        for ax in axes:
            ax.set_ylim(0, ylim)


def plot_compare_avg_time_across_conditions(
    df_to_plot: pd.DataFrame,
    x_label: str,
    y_label: str,
    y_title: str,
    save_path: str,
    cb_palette: Optional[List] = None,
    chance_perf: Optional[float] = None,
    ylim: Optional[float] = None,
):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(36, 12), sharey=True)

    plot_compare_avg_time_with_hints(
        df_to_plot,
        y_label=y_label,
        y_title=y_title,
        ax=axes[0],
        chance_perf=chance_perf,
        cb_palette=cb_palette,
        ylim=ylim,
    )

    plot_compare_avg_time_with_hints_across_stimuli(
        df_to_plot,
        y_label=y_label,
        y_title=y_title,
        axes=axes[1:],
        cb_palette=cb_palette,
        chance_perf=chance_perf,
        ylim=ylim,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center right",
        bbox_to_anchor=(1.15, 0.5),
        ncol=1,
        fontsize=44,
    )

    plt.tight_layout(rect=[0, 0, 0.98, 1])

    # Remove individual legends from subplots
    for ax in axes:
        if ax.get_legend():
            ax.get_legend().remove()

    # Adjust layout and save
    plt.tight_layout()
    os.makedirs(f"figures", exist_ok=True)
    plt.savefig(f"figures/{save_path}", bbox_inches="tight", dpi=150)
    plt.savefig(f"figures/{save_path}.pdf", format="pdf", bbox_inches="tight")


def main():

    analysis_df = pd.read_csv(f"filtered_avg_analysis_data.csv")
    df_to_plot = analysis_df.copy()
    setup_plotting()

    time_col = "quant_ques_af_exp_time_avg"
    save_path = "rqs_time_plot"

    plot_compare_avg_time_across_conditions(
        df_to_plot,
        save_path=save_path,
        x_label="part_lbl",
        y_label=time_col,
        y_title=r"Q2 -- Q4 Average Time (Minutes)",
        chance_perf=None,
        cb_palette=cb_palette,
        ylim=2.2,
    )


if __name__ == "__main__":
    main()
