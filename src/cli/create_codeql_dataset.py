from pathlib import Path

import click

from codeql.project import CodeQLProject
from codeql.utils import create_codeql_dataset
from codeql.wrapper import CodeQL

from .settings import BUILD_DIR, CODEQL_DIR

CODEQL = CodeQL(Path("codeql"))


@click.command()
@click.option("--language")
def main(language: str):
    from concurrent.futures import ProcessPoolExecutor

    target_dir = CODEQL_DIR / language / "ql" / "src" / "Security"

    def _create_codeql_dataset(project: CodeQLProject):
        return create_codeql_dataset(project, CODEQL, language)

    projects = CodeQLProject.from_directory(target_dir, BUILD_DIR / "codeql" / language)

    with ProcessPoolExecutor(max_workers=16) as executor:
        list(executor.map(_create_codeql_dataset, projects))


if __name__ == "__main__":
    main()
