from dataclasses import dataclass
from pathlib import Path

from sarif.models import StaticAnalysisResultsFormatSarifVersion210JsonSchema

CodeQLQueryResult = StaticAnalysisResultsFormatSarifVersion210JsonSchema


@dataclass
class CodeQLQuery:
    name: str
    path: Path
