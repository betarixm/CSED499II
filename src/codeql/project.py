from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Iterable, List

from .query import CodeQLQuery, CodeQLQueryResult


class CodeQLProject:
    def __init__(self, name: str, path: Path, build_dir: Path) -> None:
        self.name = name
        self.path = path
        self.build_dir = build_dir

        os.makedirs(self.build_at() / "results", exist_ok=True)

    def build_at(self) -> Path:
        return self.build_dir / self.name

    def database(self):
        return self.build_at() / "database"

    def queries(self):
        return [
            CodeQLQuery(name=Path(path).stem, path=Path(path))
            for path in glob.glob(str(self.path / "*.ql"))
        ]

    def query_result_path(self, query: CodeQLQuery):
        return self.build_at() / "results" / f"{query.name}.sarif"

    def query_result_pathes(self) -> List[Path]:
        return [
            Path(p) for p in glob.glob(str(self.build_at() / "results" / f"*.sarif"))
        ]

    def get_query_result_by_query(self, query: CodeQLQuery):
        return CodeQLQueryResult.model_validate_json(
            open(self.query_result_path(query)).read()
        )

    def get_query_results(
        self,
    ) -> Iterable[CodeQLQueryResult]:
        for result_path in self.query_result_pathes():
            yield CodeQLQueryResult.model_validate_json(open(result_path).read())

    @staticmethod
    def from_directory(projects_dir: Path, build_dir: Path) -> List[CodeQLProject]:
        return [
            CodeQLProject(name=Path(path).name, path=Path(path), build_dir=build_dir)
            for path in glob.glob(f"{projects_dir}/*")
        ]
