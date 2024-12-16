import os
import pandas as pd
import numpy as np
from scipy.stats import sem, ranksums, ks_2samp
from itertools import product
from typing import List, Optional

from utils import *


def get_plot_data(
    df_to_plot: pd.DataFrame,
    y_label: str,
    part: str,
    stimuli: str,
):
    filtered_data = df_to_plot[
        (df_to_plot["stimulus_presentation"] == stimuli)
        & (df_to_plot["part_lbl"] == part)
    ]

    stats = {
        hint_type: {
            "scores": filtered_data[filtered_data["hint_type"] == hint_type][y_label]
        }
        for hint_type in HINT_ORDER
    }

    for hint_type in stats:
        stats[hint_type].update(
            {
                "mean": stats[hint_type]["scores"].mean(),
                "sem": sem(stats[hint_type]["scores"]),
            }
        )

    p_values = {}
    none_scores = stats["No Hint"]["scores"]
    for hint_type in stats:
        if hint_type != "none":
            _, p_value_ks = ks_2samp(none_scores, stats[hint_type]["scores"])
            print(f"{stimuli}-{part}-{hint_type}")
            if p_value_ks > 0.05:
                p_val = ranksums(
                    none_scores, stats[hint_type]["scores"], alternative="less"
                ).pvalue
                print(
                    f"\tWilcoxon rank-sum test for '{part}-{stimuli}-{hint_type}': {p_val}"
                )
                if p_val > 0.05:
                    p_val = ranksums(
                        none_scores, stats[hint_type]["scores"], alternative="greater"
                    ).pvalue
                p_values[hint_type] = p_val
            else:
                p_values[hint_type] = 999

            effect_size = cohen_d(stats[hint_type]["scores"], none_scores)
            print(f"\tCohen's d for '{hint_type}': {effect_size}")
            print(
                f"\tDifference of means: {np.mean(stats[hint_type]['scores'])-np.mean(none_scores)}"
            )

    # Prepare data for plotting
    plot_data = pd.DataFrame(
        {
            "hint": HINT_ORDER,
            "mean": [stats[hint_type]["mean"] for hint_type in HINT_ORDER],
            "SEM": [stats[hint_type]["sem"] for hint_type in HINT_ORDER],
            "p-values": [1]
            + [p_values.get(hint_type, 1) for hint_type in HINT_ORDER[1:]],
        }
    )

    return plot_data


def plot_compare_hint_types_across_stimuli_participants(
    df_to_plot: pd.DataFrame,
    x_label: str,
    y_label: str,
    y_title: str,
    save_path: str,
    corr_filt: bool = False,
    cb_palette: Optional[List] = None,
    chance_perf: Optional[float] = None,
    ylim: Optional[float] = None,
):
    fig, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(32, 24), sharex="col", sharey="row"
    )

    if corr_filt:
        df_to_plot = df_to_plot[
            (df_to_plot["score_quant2"] == 1)
            & (df_to_plot["score_quant3"] == 1)
            & (df_to_plot["score_quant4"] == 1)
        ].copy()

    for i, (part, stimuli) in enumerate(product(PART_ORDER, STIMULUS_ORDER)):

        row = i // 2
        col = i % 2
        ax = axes[row, col]

        plot_df = get_plot_data(df_to_plot, y_label, part, stimuli)

        no_hint_data = plot_df[plot_df["hint"] == "No Hint"].iloc[0]
        hint_data = plot_df[plot_df["hint"] != "No Hint"]

        for i, (_, row_hint) in enumerate(hint_data.iterrows()):

            if row_hint["mean"] - no_hint_data["mean"] >= 0:
                # Plot 'No Hint' bar
                ax.bar(
                    i,
                    no_hint_data["mean"],
                    color=cb_palette[0] if stimuli == "Python" else cb_palette[1],
                    edgecolor="black",
                )

                # Plot 'Any Hint' bar on top
                bar = ax.bar(
                    i,
                    row_hint["mean"] - no_hint_data["mean"],
                    bottom=no_hint_data["mean"],
                    color=cb_palette[0] if stimuli == "Python" else cb_palette[1],
                    alpha=0.5,
                    hatch="//",
                    edgecolor="black",
                )

            else:

                # Plot 'No Hint' bar on top
                bar = ax.bar(
                    i,
                    no_hint_data["mean"] - row_hint["mean"],
                    bottom=row_hint["mean"],
                    color=cb_palette[0] if stimuli == "Python" else cb_palette[1],
                    edgecolor="black",
                )

                # Plot 'Any Hint' bar
                ax.bar(
                    i,
                    row_hint["mean"],
                    color=cb_palette[0] if stimuli == "Python" else cb_palette[1],
                    alpha=0.5,
                    hatch="//",
                    edgecolor="black",
                )

            ax.errorbar(
                x=i,
                y=row_hint["mean"],
                yerr=row_hint["SEM"],
                fmt="none",
                color="black",
                capsize=5,
            )
            ax.errorbar(
                x=i,
                y=no_hint_data["mean"],
                yerr=no_hint_data["SEM"],
                fmt="none",
                color="black",
            )

            # Add p-value asterisks
            if row_hint["p-values"] < 0.05:
                p_val_pos = 4.8 if corr_filt else 0.6
                ax.text(
                    i,
                    p_val_pos,
                    get_asterisks(row_hint["p-values"]),
                    ha="center",
                    va="bottom",
                    color="black",
                )

        ax.set_xticks(range(len(hint_data)))
        ax.set_xticklabels(hint_data["hint"])
        ax.set_title(f"{stimuli.capitalize()} + {part.capitalize()}")

        if col == 0:
            ax.set_ylabel(y_title, labelpad=20)
        else:
            ax.set_ylabel("")

        # Rotate x-ticks for all subplots
        if row == 1:
            ax.set_xlabel("")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

        if chance_perf is not None:
            ax.axhline(y=chance_perf, color="r", linestyle="--")

    # Create legend elements
    legend_elements = [
        plt.Rectangle(
            (0, 0), 1, 1, facecolor=cb_palette[0], edgecolor="black", label="Python"
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=cb_palette[0],
            alpha=0.5,
            hatch="//",
            edgecolor="black",
            label="Python + Hint",
        ),
        plt.Rectangle(
            (0, 0), 1, 1, facecolor=cb_palette[1], edgecolor="black", label="Text"
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=cb_palette[1],
            alpha=0.5,
            hatch="//",
            edgecolor="black",
            label="Text + Hint",
        ),
    ]

    # Set y-limit for all subplots
    if ylim:
        plt.setp(axes, ylim=(0, ylim))

    fig.legend(
        handles=legend_elements,
        loc="center right",
        ncol=1,
        bbox_to_anchor=(1.22, 0.5),
        fontsize=58,
    )

    plt.tight_layout(rect=[0, 0, 0.98, 1])

    os.makedirs(f"figures", exist_ok=True)
    plt.savefig(f"figures/{save_path}", bbox_inches="tight", dpi=75)
    plt.savefig(
        f"figures/{save_path}.pdf",
        format="pdf",
        bbox_inches="tight",
        dpi=75,
    )


def main():
    analysis_df = pd.read_csv(f"avg_analysis_data.csv")
    df_to_plot = analysis_df.copy()

    setup_plotting(scale=3)

    plot_compare_hint_types_across_stimuli_participants(
        df_to_plot,
        save_path="rq3_eff_hint_type_acc",
        x_label="part_lbl",
        y_label="quant234_avg_score",
        y_title=r"Q2 -- Q4 Avgerage Accuracy",
        chance_perf=0.30,
        ylim=0.8,
        cb_palette=cb_palette,
    )


if __name__ == "__main__":
    main()
