from utils import *

import os
import pandas as pd
import numpy as np
from scipy.stats import sem, ranksums, ks_2samp
from itertools import product
from typing import List, Optional


def get_plot_data(
    df_to_plot: pd.DataFrame,
    y_label: str,
):
    combined_hints = {"Any Hint": HINT_ORDER[1:]}

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

    p_values = {}
    for stim_pres, part_lbl in product(STIMULUS_ORDER, PART_ORDER):
        none_scores = stats[("No Hint", stim_pres, part_lbl)]["scores"]
        hint_scores = stats[("Any Hint", stim_pres, part_lbl)]["scores"]

        _, p_value_ks = ks_2samp(none_scores, hint_scores)

        if p_value_ks > 0.05:
            p_val = ranksums(none_scores, hint_scores, alternative="less").pvalue
            print(f"Ranksum P-Value: {p_val}")
            if p_val > 0.05:
                p_val = ranksums(none_scores, hint_scores, alternative="greater").pvalue
            p_values[(stim_pres, part_lbl)] = p_val
        else:
            p_values[(stim_pres, part_lbl)] = 999

        ## Print effect size
        effect_size = cohen_d(hint_scores, none_scores)
        print(f"\tCohen's d for '{stim_pres}-{part_lbl}': {effect_size}")
        print(f"\tDifference of Means: {mean(hint_scores) - mean(none_scores)}")

    plot_data = []
    for hint_type, stim_pres, part_lbl in product(
        ["No Hint", "Any Hint"], STIMULUS_ORDER, PART_ORDER
    ):
        key = (hint_type, stim_pres, part_lbl)
        plot_data.append(
            {
                "hint_type": hint_type,
                "stim_pres": stim_pres,
                "part_lbl": part_lbl,
                "mean": stats[key]["mean"],
                "sem": stats[key]["sem"],
                "p-value": (
                    1
                    if hint_type == "No Hint"
                    else p_values.get((stim_pres, part_lbl), 1)
                ),
            }
        )
    plot_df = pd.DataFrame(plot_data)
    return plot_df


def plot_compare_hints_across_stimuli_participants(
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

    if corr_filt:
        df_to_plot = df_to_plot[
            (df_to_plot["score_quant2"] == 1)
            & (df_to_plot["score_quant3"] == 1)
            & (df_to_plot["score_quant4"] == 1)
        ].copy()

    plot_df = get_plot_data(df_to_plot, y_label)

    fig, ax = plt.subplots(figsize=(16, 12))
    bar_width = 0.35
    r1 = np.arange(2)
    r2 = [x + bar_width for x in r1]

    for i, stimulus in enumerate(STIMULUS_ORDER):
        for j, part_lbl in enumerate(PART_ORDER):
            none_value = plot_df[
                (plot_df["hint_type"] == "No Hint")
                & (plot_df["stim_pres"] == stimulus)
                & (plot_df["part_lbl"] == part_lbl)
            ]["mean"].values[0]
            none_sem = plot_df[
                (plot_df["hint_type"] == "No Hint")
                & (plot_df["stim_pres"] == stimulus)
                & (plot_df["part_lbl"] == part_lbl)
            ]["sem"].values[0]

            any_exp_value = plot_df[
                (plot_df["hint_type"] == "Any Hint")
                & (plot_df["stim_pres"] == stimulus)
                & (plot_df["part_lbl"] == part_lbl)
            ]["mean"].values[0]
            any_exp_sem = plot_df[
                (plot_df["hint_type"] == "Any Hint")
                & (plot_df["stim_pres"] == stimulus)
                & (plot_df["part_lbl"] == part_lbl)
            ]["sem"].values[0]

            p_value = plot_df[
                (plot_df["hint_type"] == "Any Hint")
                & (plot_df["stim_pres"] == stimulus)
                & (plot_df["part_lbl"] == part_lbl)
            ]["p-value"].values[0]

            ax.bar(
                r1[j] + i * bar_width,
                none_value,
                width=bar_width,
                color=cb_palette[i],
                label=f"{stimulus}" if j == 0 else "",
                yerr=none_sem,
                edgecolor="black",
            )

            bar = ax.bar(
                r1[j] + i * bar_width,
                any_exp_value - none_value,
                width=bar_width,
                bottom=none_value,
                color=cb_palette[i],
                label=f"{stimulus} + Hint" if j == 0 else "",
                hatch="//",
                alpha=0.5,
                yerr=any_exp_sem,
                edgecolor="black",
            )

            # Add significance marker if p-value is significant
            if p_value < 0.05:
                ax.text(
                    r1[j] + i * bar_width,
                    0.6,
                    get_asterisks(p_value),
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

    ax.set_ylabel(y_title, labelpad=15)
    ax.set_xticks([r + bar_width / 2 for r in r1])
    ax.set_xticklabels(PART_ORDER)

    ax.legend(loc="best", bbox_to_anchor=(0.94, 1.30), ncol=2)

    if ylim:
        plt.ylim(0, 0.8)
    if chance_perf:
        plt.axhline(y=chance_perf, color="r", linestyle="--")

    plt.tight_layout()
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

    setup_plotting()

    plot_compare_hints_across_stimuli_participants(
        df_to_plot,
        save_path="rq2_eff_hint_acc",
        x_label="part_lbl",
        y_label="quant234_avg_score",
        y_title=r"Q2 -- Q4 Avgerage Accuracy",
        chance_perf=0.30,
        ylim=0.8,
        cb_palette=cb_palette,
    )


if __name__ == "__main__":
    main()
