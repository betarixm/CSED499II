from itertools import chain
from typing import List

from .models import Location, StaticAnalysisResultsFormatSarifVersion210JsonSchema


def query_result_to_sarif_results(
    result: StaticAnalysisResultsFormatSarifVersion210JsonSchema,
):
    return chain(*[run.results for run in result.runs if run.results is not None])


def extract_regions_with_uri_from_sarif_locations(
    locations: List[Location] | None = None,
):
    if locations is None:
        locations = []

    return [
        (
            location.physical_location.region,
            location.physical_location.artifact_location.uri,
        )
        for location in locations
        if location.physical_location is not None
        and location.physical_location.region is not None
        and location.physical_location.artifact_location is not None
        and location.physical_location.artifact_location.uri is not None
    ]
