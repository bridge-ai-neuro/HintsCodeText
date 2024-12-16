from utils import *

import pandas as pd
import numpy as np
from scipy.stats import sem, ranksums, ks_2samp
from typing import List, Optional


def categorize_exp(x):
    if 0 <= x <= 1:
        return "Low"
    elif 1 < x < 3:
        return "Medium"
    elif 3 <= x <= 5:
        return "High"
    else:
        return "Other"


def plot_compare_prog_exp_across_stimuli(
    df_to_plot,
    save_path="rq1_eff_pres_mod_acc",
    x_label="part_lbl",
    y_label="quant234_avg_score",
    y_title=r"Q2 -- Q4 Avgerage Accuracy",
    chance_perf=0.305,
    ylim=0.8,
    cb_palette=cb_palette,
    p_value_pos: float = 0.6,
    p_value_len1: float = 0.05,
    p_value_len2: float = 0.1,
    text_buffer: float = 0.105,
):
    df_to_plot["exp_group"] = df_to_plot["Q_prog_exp"].apply(categorize_exp)
    plot_df = df_to_plot[df_to_plot["exp_group"].isin(["Low", "Medium", "High"])].copy()

    fig, ax = plt.subplots(figsize=(16, 12))
    bar_width = 0.35
    p_values = {}

    x_labels = ["Low", "Medium", "High"]
    x_tickers = np.arange(len(x_labels))

    for i, stimulus in enumerate(STIMULUS_ORDER):
        data = [
            plot_df[
                (plot_df[x_label] == label)
                & (plot_df["stimulus_presentation"] == stimulus)
            ][y_label].mean()
            for label in x_labels
        ]

        errors = [
            plot_df[
                (plot_df[x_label] == label)
                & (plot_df["stimulus_presentation"] == stimulus)
            ][y_label].sem()
            for label in x_labels
        ]

        ax.bar(
            x_tickers + i * bar_width - bar_width / 2,
            data,
            bar_width,
            label=stimulus,
            color=cb_palette[i],
            yerr=errors,
            edgecolor="black",
        )

    for label in x_labels:
        sub_df = plot_df[plot_df[x_label] == label]

        text_scores = sub_df[sub_df["stimulus_presentation"] == "Text"][y_label]
        python_scores = sub_df[sub_df["stimulus_presentation"] == "Python"][y_label]
        print(f"{len(sub_df)} participants in {label}")
        print(f"{len(text_scores)} participants in Text")
        print(f"{len(python_scores)} participants in Python")

        _, p_value_ks = ks_2samp(python_scores, text_scores)
        if p_value_ks > 0.05:
            _, p_value = ranksums(python_scores, text_scores, alternative="less")
            p_values[label] = p_value
        else:
            p_values[label] = 999

        print(f"p-value for {label}: {p_value}")
        print(f"\tCohen's d for {label}: {cohen_d(text_scores, python_scores)}")
        print(f"\tDifference of means: {np.mean(text_scores)-np.mean(python_scores)}")

    ax.set_xlabel("")
    ax.set_ylabel(y_title, labelpad=15)
    ax.set_xticks(x_tickers)
    ax.set_xticklabels(x_labels)
    ax.grid(False)

    if chance_perf is not None:
        ax.axhline(y=chance_perf, color="r", linestyle="--")

    for i, label in enumerate(x_labels):
        if p_values[label] <= 0.05:
            x1 = i - 0.2
            x2 = i + 0.2

            ax.plot(
                [x1, x1, x2, x2],
                [
                    p_value_pos + p_value_len1,
                    p_value_pos + p_value_len2,
                    p_value_pos + p_value_len2,
                    p_value_pos + p_value_len1,
                ],
                c="black",
            )

            stars = get_asterisks(p_values[label])

            ax.text(
                (x1 + x2) * 0.5,
                p_value_pos + text_buffer,
                f"{stars}",
                ha="center",
                va="bottom",
                color="black",
            )

    ax.legend(loc="best", bbox_to_anchor=(0.78, 1.15), ncol=2)
    if ylim is not None:
        ax.set_ylim(0, ylim)

    plt.tight_layout()
    plt.savefig(f"figures/{save_path}", bbox_inches="tight", dpi=75)
    plt.savefig(f"figures/{save_path}.pdf", format="pdf", bbox_inches="tight", dpi=75)

    print(p_values)
    return p_values


def main():
    ## Load the Dataframe
    analysis_df = pd.read_csv("avg_analysis_data.csv")
    df_to_plot = analysis_df[
        (analysis_df["hint_type"] == "No Hint") & (analysis_df["part_lbl"] == "Clear")
    ]

    setup_plotting()
    plot_compare_prog_exp_across_stimuli(
        df_to_plot,
        save_path="h1_v_h2_analysis",
        x_label="exp_group",
        y_label="quant234_avg_score",
        y_title=r"Q2 -- Q4 Avgerage Accuracy",
        chance_perf=0.305,
        ylim=0.8,
        cb_palette=cb_palette,
    )


if __name__ == "__main__":
    main()
