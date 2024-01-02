import click


@click.command()
@click.option("--model")
@click.option("--len-soft-prompt", required=False)
@click.option("--soft-prompt-path", required=False)
def main(model: str, len_soft_prompt: int | None, soft_prompt_path: str | None):
    from pathlib import Path

    from datasets import Dataset  # type: ignore
    from torch.utils.data import DataLoader

    from analyzer.wrapper import Analyzer
    from cli.settings import BUILD_DIR, DEPENDENCIES_DIR
    from dataset.models import TrainingData

    def data_generator_factory(path: Path):
        def _():
            for data in TrainingData.from_csv(path):
                yield data.model_dump()

        return _

    test_dataset = Dataset.from_generator(  # type: ignore
        data_generator_factory(
            BUILD_DIR / "dataset-test.csv",
        ),
    )

    test_dataloader = DataLoader[TrainingData](
        test_dataset,  # type: ignore
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: x,  # type: ignore
        pin_memory=True,
    )

    analyzer = Analyzer(
        DEPENDENCIES_DIR / model,
        ["cuda"],
        len_soft_prompt=len_soft_prompt,
        soft_prompt_path=DEPENDENCIES_DIR / soft_prompt_path
        if soft_prompt_path
        else None,
    )

    output_file = open(
        BUILD_DIR
        / f"line-probabilities-{model}-{str(soft_prompt_path).replace('/', '-')}.csv".lower(),
        "w",
    )

    analyzer.evaluate(test_dataloader, output_file)


if __name__ == "__main__":
    main()
