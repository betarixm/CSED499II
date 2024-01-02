from .project import CodeQLProject
from .wrapper import CodeQL


def create_codeql_dataset(project: CodeQLProject, codeql: CodeQL, language: str):
    codeql.create_database(project.database(), language, project.path)
    codeql.finalize_database(project.database())

    return project.database()


def analyze_codeql_dataset(project: CodeQLProject, codeql: CodeQL):
    for query in project.queries():
        codeql.analyze_database(
            project.database(),
            query.path,
            "sarif-latest",
            project.query_result_path(query),
        )

    return [project.query_result_path(query) for query in project.queries()]
