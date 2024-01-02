import random
from itertools import groupby
from typing import Iterable

from cli.settings import BUILD_DIR
from dataset.models import (
    CodeWithVulnerableLineAnnotation,
    Location,
    MergedCode,
    TrainingData,
)


def _interleave_data(
    data: Iterable[TrainingData], inverval: int = 10
) -> Iterable[TrainingData]:
    not_vulnerable_index = 0

    for datum in data:
        if datum.is_vulnerable:
            yield datum
        else:
            if not_vulnerable_index % inverval == 0:
                yield datum

            not_vulnerable_index += 1


def main():
    csvs = [csv for csv in BUILD_DIR.glob("code-with-vulnerable-line-annotation-*.csv")]

    codes = [
        code for csv in csvs for code in CodeWithVulnerableLineAnnotation.from_csv(csv)
    ]

    grouped = groupby(sorted(codes, key=lambda code: code.code), lambda code: code.code)

    merged_codes = [
        MergedCode(
            code=code,
            locations=[
                Location(
                    start_line=code.start_line,
                    end_line=code.end_line,
                    rule_id=code.rule_id,
                    start_column=code.start_column,
                    end_column=code.end_column,
                )
                for code in codes
            ],
        )
        for code, codes in grouped
    ]

    data = [_ for merged_code in merged_codes for _ in merged_code.to_training_data()]
    data = [_ for _ in data if _.value != "" and _.prefix != ""]

    data = list(_interleave_data(data))

    random.seed(1337)
    random.shuffle(data)

    train_data = data[: int(len(data) * 0.8)]
    test_data = data[int(len(data) * 0.8) :]

    train_file = open(BUILD_DIR / "dataset-train.csv", "w")
    test_file = open(BUILD_DIR / "dataset-test.csv", "w")

    train_writer = TrainingData.use_csv_dict_writer_with_file(train_file)
    test_writer = TrainingData.use_csv_dict_writer_with_file(test_file)

    for row in train_data:
        row.write_csv_row(train_writer)

    for row in test_data:
        row.write_csv_row(test_writer)

    print("[*] Train:")
    print(f"    - Total: {len(train_data)}")
    print(f"    - Vul: {len([_ for _ in train_data if _.is_vulnerable])}")
    print(f"    - Not Vul: {len([_ for _ in train_data if not _.is_vulnerable])}")

    print("[*] Test:")
    print(f"    - Total: {len(test_data)}")
    print(f"    - Vul: {len([_ for _ in test_data if _.is_vulnerable])}")
    print(f"    - Not Vul: {len([_ for _ in test_data if not _.is_vulnerable])}")


if __name__ == "__main__":
    main()
