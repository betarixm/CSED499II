import click


@click.command()
@click.option("--model")
@click.option("--prefix-tuned", required=False)
@click.option("--batch-size", default=1)
@click.option("--num-epochs", default=8)
@click.option("--lr", default=1e-9)
@click.option("--max-norm", default=5)
@click.option("--accumulate", default=1)
@click.option("--len-soft-prompt", default=8)
@click.option("--until", default=None, required=False, type=int)
def main(
    model: str,
    prefix_tuned: str | None,
    batch_size: int,
    num_epochs: int,
    lr: float,
    max_norm: float,
    accumulate: int,
    len_soft_prompt: int,
    until: int | None = None,
):
    from pathlib import Path

    from datasets import Dataset  # type: ignore
    from torch.utils.data import DataLoader

    from analyzer.wrapper import Analyzer, TrainingData
    from cli.settings import BUILD_DIR, DEPENDENCIES_DIR
    from dataset.models import TrainingData

    def data_generator_factory(path: Path):
        def _():
            for data in TrainingData.from_csv(path):
                yield data.model_dump()

        return _

    train_dataset = Dataset.from_generator(  # type: ignore
        data_generator_factory(
            BUILD_DIR / "dataset-train.csv",
        ),
    )

    test_dataset = Dataset.from_generator(  # type: ignore
        data_generator_factory(
            BUILD_DIR / "dataset-test.csv",
        ),
    )

    train_dataloader = DataLoader[TrainingData](
        train_dataset,  # type: ignore
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda x: x,  # type: ignore
        pin_memory=True,
    )

    test_dataloader = DataLoader[TrainingData](
        test_dataset,  # type: ignore
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda x: x,  # type: ignore
        pin_memory=True,
    )

    analyzer = Analyzer(
        DEPENDENCIES_DIR / model,
        ["cuda"],
        len_soft_prompt=len_soft_prompt,
        soft_prompt_path=DEPENDENCIES_DIR / prefix_tuned if prefix_tuned else None,
    )

    analyzer.train(
        train_dataloader,
        test_dataloader,
        lr=lr,
        num_epochs=num_epochs,
        batch_size=batch_size,
        accumulate_grad_batches=accumulate,
        max_norm=max_norm,
        until=until,
    )


if __name__ == "__main__":
    main()
