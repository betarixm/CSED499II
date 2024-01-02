from typing import List, Tuple

import click
import matplotlib.pyplot as plt
import seaborn as sns

from dataset.models import LineProbability

from .settings import BUILD_DIR


def is_included(
    line_probabilities: List[LineProbability], line_probability: LineProbability
) -> bool:
    for p in line_probabilities:
        if (
            p.line == line_probability.line
            and p.is_vulnerable == line_probability.is_vulnerable
            and p.probability == line_probability.probability
            and p.model == line_probability.model
            and p.line_number == line_probability.line_number
            and p.probability >= -600
        ):
            return True

    return False


def fpr_and_tpr_by_threshold(results: List[Tuple[bool, float]], threshold: float):
    predictions: List[Tuple[bool, float, bool]] = [
        (not is_vulnerable, probability, probability > threshold)
        for is_vulnerable, probability in results
    ]

    false_positives = len(
        [_ for truth, _, prediction in predictions if not truth and prediction]
    )

    true_positives = len(
        [_ for truth, _, prediction in predictions if truth and prediction]
    )

    falsies = len([_ for truth, _, _ in predictions if not truth])

    trues = len([_ for truth, _, _ in predictions if truth])

    return false_positives / falsies, true_positives / trues


def auc(fpr_and_tpr: List[Tuple[float, float]]):
    sorted_fpr_and_tpr = sorted(fpr_and_tpr, key=lambda x: x[0])

    return sum(
        [
            (sorted_fpr_and_tpr[i + 1][0] - sorted_fpr_and_tpr[i][0])
            * (sorted_fpr_and_tpr[i + 1][1] + sorted_fpr_and_tpr[i][1])
            / 2
            for i in range(len(sorted_fpr_and_tpr) - 1)
        ]
    )


@click.command()
@click.option("--model")
def main(model: str):
    line_probabilities: List[LineProbability] = []

    for line_probability in list(
        LineProbability.from_csv(BUILD_DIR / f"line-probabilities-{model}.csv")
    ):
        if is_included(line_probabilities, line_probability):
            continue

        if line_probability.line == "":
            continue

        line_probabilities.append(line_probability)

    fpr_and_tpr = [
        fpr_and_tpr_by_threshold(
            [(p.is_vulnerable, p.probability) for p in line_probabilities], threshold
        )
        for threshold in range(-600, 0)
    ]

    print(auc(fpr_and_tpr))

    sns.set_theme(style="whitegrid", font="Pretendard", font_scale=1)  # type: ignore

    fig, ax = plt.subplots()  # type: ignore

    fig.set_size_inches(5.25, 3.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")

    sns.lineplot(
        x=[fpr for fpr, _ in fpr_and_tpr],
        y=[tpr for _, tpr in fpr_and_tpr],
        ax=ax,
        color="#344877",
    )

    sns.lineplot(
        x=[0, 1],
        y=[0, 1],
        ax=ax,
        color="grey",
        linewidth=0.5,
    )

    plt.savefig(  # type: ignore
        BUILD_DIR / f"roc-{model}.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
    )


if __name__ == "__main__":
    main()
