import shutil
import subprocess
from pathlib import Path


class CodeQL:
    def __init__(self, executable: Path) -> None:
        self._executable: Path = executable

    def create_database(
        self, database: Path, language: str, source_root: Path, cwd: Path | None = None
    ):
        shutil.rmtree(database, ignore_errors=True)

        return subprocess.run(
            [
                self._executable,
                "database",
                "create",
                database,
                "--language",
                language,
                "--source-root",
                source_root,
            ],
            cwd=cwd,
        )

    def analyze_database(self, database: Path, query: Path, format_: str, output: Path):
        return subprocess.run(
            [
                self._executable,
                "database",
                "analyze",
                database,
                query,
                "--format",
                format_,
                "--output",
                output,
            ],
            capture_output=True,
        )

    def finalize_database(self, database: Path):
        return subprocess.run([self._executable, "database", "finalize", database])
