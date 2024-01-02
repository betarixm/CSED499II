from typing import List

import click
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore

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
        ):
            return True

    return False


@click.command()
@click.option("--model")
@click.option("--binwidth", type=int)
def main(
    model: str,
    binwidth: int,
):
    line_probabilities: List[LineProbability] = []

    for line_probability in list(
        LineProbability.from_csv(BUILD_DIR / f"line-probabilities-{model}.csv")
    ):
        if is_included(line_probabilities, line_probability):
            continue

        if line_probability.line == "":
            continue

        line_probabilities.append(line_probability)

    # Show the number of lines that are vulnerable and not vulnerable

    vulnerable_lines = len([p for p in line_probabilities if p.is_vulnerable])

    not_vulnerable_lines = len([p for p in line_probabilities if not p.is_vulnerable])

    data = {
        "Line Probability": [p.probability for p in line_probabilities],
        "is_vulnerable": [
            f"Vulnerable (n={vulnerable_lines})"
            if p.is_vulnerable
            else f"Not Vulnerable (n={not_vulnerable_lines})"
            for p in line_probabilities
        ],
    }

    sns.set_theme(style="whitegrid", font="Pretendard", font_scale=1)  # type: ignore
    fig, ax = plt.subplots()  # type: ignore

    fig.set_size_inches(5, 3.5)
    ax.set_xlim(-600, 0)
    ax.set_ylim(0, 14)

    ax = sns.histplot(  # type: ignore
        data=data,
        x="Line Probability",
        kde=True,
        hue="is_vulnerable",
        palette="cool",
        alpha=0.5,
        binwidth=binwidth,  # type: ignore
        stat="percent",
        common_norm=False,
        ax=ax,
    )

    # ax.set_title(f"Line Probabilities of {model}")  # type: ignore
    ax.get_legend().set_title("")  # type: ignore

    plt.tight_layout()

    plt.savefig(BUILD_DIR / f"line-probabilities-{model}.png", dpi=600, bbox_inches="tight", pad_inches=0, transparent=True)  # type: ignore


if __name__ == "__main__":
    main()
