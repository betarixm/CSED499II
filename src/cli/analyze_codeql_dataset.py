from pathlib import Path

import click
from tqdm import tqdm

from codeql.project import CodeQLProject
from codeql.utils import analyze_codeql_dataset
from codeql.wrapper import CodeQL

from .settings import BUILD_DIR, CODEQL_DIR

CODEQL = CodeQL(Path("codeql"))


@click.command()
@click.option("--language")
def main(language: str):
    from concurrent.futures import ProcessPoolExecutor

    target_dir = CODEQL_DIR / language / "ql" / "src" / "Security"

    def _analyze_codeql_dataset(project: CodeQLProject):
        result = analyze_codeql_dataset(project, CODEQL)
        return result

    projects = CodeQLProject.from_directory(target_dir, BUILD_DIR / "codeql" / language)

    with ProcessPoolExecutor(max_workers=16) as executor:
        list(
            tqdm(
                executor.map(_analyze_codeql_dataset, projects),
                total=len(projects),
            ),
        )


if __name__ == "__main__":
    main()
