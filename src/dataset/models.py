from __future__ import annotations

import base64
import hashlib
import json
from abc import abstractmethod
from csv import DictReader, DictWriter
from pathlib import Path
from typing import Annotated, Any, Collection, Iterable, List, TextIO

from pydantic import (
    BaseModel,
    TypeAdapter,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    computed_field,
)
from pydantic.functional_serializers import PlainSerializer
from pydantic.functional_validators import WrapValidator

from codeql.project import CodeQLProject
from codeql.query import CodeQLQueryResult
from sarif.models import Result as SarifResult
from sarif.utils import extract_regions_with_uri_from_sarif_locations


def _maybe_base64_encoded_str(
    v: Any, handler: ValidatorFunctionWrapHandler, info: ValidationInfo
):
    if info.mode == "json":
        return handler(base64.b64decode(str(v)))
    else:
        return v


Base64Str = Annotated[
    str,
    PlainSerializer(
        lambda s: base64.b64encode(str(s).encode()).decode(),
        return_type=str,
        when_used="json",
    ),
    WrapValidator(_maybe_base64_encoded_str),
]


class BaseModelWithCsv(BaseModel):
    @staticmethod
    @abstractmethod
    def fieldnames() -> Collection[str]:
        pass

    @classmethod
    def from_csv(cls, path: Path):
        reader = DictReader(open(path))

        for row in reader:
            yield cls.model_validate_json(
                json.dumps({key: value for key, value in row.items() if value != ""})
            )

    @classmethod
    def use_csv_dict_writer(cls, path: Path):
        writer = DictWriter(
            open(path, "w"),
            fieldnames=cls.fieldnames(),
        )

        writer.writeheader()

        return writer

    @classmethod
    def use_csv_dict_writer_with_file(cls, file: TextIO):
        writer = DictWriter(
            file,
            fieldnames=cls.fieldnames(),
        )

        writer.writeheader()

        return writer

    def write_csv_row(self, writer: DictWriter[str]):
        writer.writerow(self.model_dump(mode="json"))


class CodeWithVulnerableLineAnnotation(BaseModelWithCsv):
    rule_id: str
    code: Base64Str
    start_line: int
    start_column: int | None = None
    end_line: int | None = None
    end_column: int | None = None

    def __hash__(self) -> str:
        return hashlib.sha256(
            f"{self.rule_id}{self.code}{self.start_line}{self.start_column}{self.end_column}{self.end_line}".encode()
        ).hexdigest()

    @computed_field
    @property
    def id(self) -> str:
        return self.__hash__()

    @staticmethod
    def fieldnames() -> Collection[str]:
        return [
            "id",
            *CodeWithVulnerableLineAnnotation.model_fields.keys(),
        ]

    @staticmethod
    def from_sarif_result(
        sarif_result: SarifResult, project: CodeQLProject
    ) -> Iterable[CodeWithVulnerableLineAnnotation]:
        if sarif_result.rule_id is None:
            raise ValueError(
                f"rule_id is None. (sarif_result: {sarif_result}, project: {project})"
            )

        for region, uri in extract_regions_with_uri_from_sarif_locations(
            sarif_result.locations
        ):
            if region.start_line is None:
                raise ValueError(
                    f"start_line is None. (sarif_result: {sarif_result}, project: {project})"
                )

            code = open(project.path / uri).read()

            yield CodeWithVulnerableLineAnnotation(
                rule_id=sarif_result.rule_id,
                code=code,
                start_line=region.start_line,
                start_column=region.start_column,
                end_line=region.end_line,
                end_column=region.end_column,
            )

    @staticmethod
    def from_query_result(
        query_result: CodeQLQueryResult, project: CodeQLProject
    ) -> Iterable[CodeWithVulnerableLineAnnotation]:
        sarif_results = [
            result
            for run in query_result.runs
            if run.results is not None
            for result in run.results
        ]

        return (
            code
            for sarif_result in sarif_results
            for code in CodeWithVulnerableLineAnnotation.from_sarif_result(
                sarif_result, project
            )
        )

    @staticmethod
    def from_project(
        project: CodeQLProject,
    ) -> Iterable[CodeWithVulnerableLineAnnotation]:
        return (
            code
            for query_result in project.get_query_results()
            for code in CodeWithVulnerableLineAnnotation.from_query_result(
                query_result, project
            )
        )

    def is_vulnerable_line(self, line_number: int):
        end_line = self.end_line if self.end_line is not None else self.start_line

        return line_number in range(self.start_line, end_line + 1)


class Location(BaseModel):
    rule_id: str
    start_line: int
    start_column: int | None = None
    end_line: int | None = None
    end_column: int | None = None

    def is_vulnerable_line(self, line_number: int):
        end_line = self.end_line if self.end_line is not None else self.start_line

        return line_number in range(self.start_line, end_line + 1)


class MergedCode(BaseModel):
    code: str
    locations: List[Location]

    def is_vulnerable_line(self, line_number: int):
        return any(
            location.is_vulnerable_line(line_number) for location in self.locations
        )

    def find_rule_id_by_line_number(self, line_number: int) -> str:
        for location in self.locations:
            if location.is_vulnerable_line(line_number):
                return location.rule_id

        return "-1"

    def to_training_data(self) -> Iterable[TrainingData]:
        lines = self.code.splitlines()

        for index, line in enumerate(lines):
            yield TrainingData(
                rule_id=self.find_rule_id_by_line_number(index + 1),
                value=line,
                prefix="\n".join(lines[max(0, index - 8) : index]),
                is_vulnerable=self.is_vulnerable_line(index + 1),
            )


class TrainingData(BaseModelWithCsv):
    rule_id: str
    value: Base64Str
    prefix: Base64Str
    is_vulnerable: bool

    @staticmethod
    def fieldnames() -> Collection[str]:
        return [
            *TrainingData.model_fields.keys(),
        ]


TrainingDataList = TypeAdapter(List[TrainingData])


class LineProbability(BaseModelWithCsv):
    line_number: int
    probability: float
    is_vulnerable: bool
    model: str
    line: Base64Str
    prefix: Base64Str
    rule_id: str

    @staticmethod
    def fieldnames() -> Collection[str]:
        return [
            *LineProbability.model_fields.keys(),
        ]

    def render(self):
        return f"{self.line.rstrip()} # Prob: {self.probability:.16f}, Vuln: {self.is_vulnerable}"

    @classmethod
    def from_csv(cls, path: Path):
        reader = DictReader(open(path))

        for row in reader:
            if row["line"] == "":
                continue

            yield LineProbability.model_validate_json(
                json.dumps({key: value for key, value in row.items() if value != ""})
            )
