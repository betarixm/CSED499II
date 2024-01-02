import click

from codeql.project import CodeQLProject
from dataset.models import CodeWithVulnerableLineAnnotation

from .settings import BUILD_DIR, CODEQL_DIR


@click.command()
@click.option("--language")
def main(language: str):
    target_dir = CODEQL_DIR / language / "ql" / "src" / "Security"

    projects = CodeQLProject.from_directory(target_dir, BUILD_DIR / "codeql" / language)

    writer = CodeWithVulnerableLineAnnotation.use_csv_dict_writer(
        BUILD_DIR / f"code-with-vulnerable-line-annotation-{language}.csv"
    )

    for code in (
        code
        for project in projects
        for code in CodeWithVulnerableLineAnnotation.from_project(project)
    ):
        code.write_csv_row(writer)


if __name__ == "__main__":
    main()
