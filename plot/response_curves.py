import numpy as np
import warnings
from matplotlib import pyplot as plt


def ibm_color_gen():
    yield from ["#648FFF", "#FE6100", "#785EF0", "#DC267F", "#FFB000"]


def plot_response_curves(stream_results, stream_name):
    plt.rcParams.update({"font.size": 18})
    fig = plt.gcf()
    fig.set_size_inches(12, 7.5)
    np.seterr("raise")
    colors = ibm_color_gen()
    for detector, results in stream_results.items():
        for key, scores in results.items():
            color = next(colors)
            if len(scores) > 1:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    padded_scores = _pad_score(scores)
                    y_values = np.nanmean(padded_scores, axis=0)
                    y_interval = 2 * np.nanstd(
                        padded_scores, axis=0
                    )  # 95% confidence interval
                lower_bound = y_values - y_interval
                clipped_lower_bound = np.clip(lower_bound, 0, 1)
                upper_bound = y_values + y_interval
                clipped_upper_bound = np.clip(upper_bound, 0, 1)
                plt.fill_between(
                    np.arange(len(y_values)),
                    clipped_lower_bound,
                    clipped_upper_bound,
                    alpha=0.2,
                    color=color,
                )
            else:
                y_values = scores[0]
            if "," in key:
                label = f"{detector}({key})"
            else:
                label = f"{detector}"
            plt.plot(y_values, label=label, linewidth=2.5, color=color)
    plt.xlabel(r"$\Delta_{\mathrm{max}}$")
    plt.ylabel(r"$F_1$")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{stream_name}.pdf", format="pdf")
    plt.clf()


def _pad_score(scores):
    max_len = max(map(len, scores))
    padded_scores = []
    for score in scores:
        score = np.array(score, dtype=float)
        padded_scores.append(
            np.pad(
                score, (0, max_len - len(score)), "constant", constant_values=(np.nan,)
            )
        )
    return padded_scores
