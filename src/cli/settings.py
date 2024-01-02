from pathlib import Path

SCRIPT_PATH = Path(__file__)
PROJECT_DIR = SCRIPT_PATH.parent.parent.parent
BUILD_DIR = PROJECT_DIR / "build"
DEPENDENCIES_DIR = PROJECT_DIR / "dependencies"
CODEQL_DIR = DEPENDENCIES_DIR / "codeql"
CODEGEN_350M_MONO_MODEL_DIR = DEPENDENCIES_DIR / "codegen-350M-mono"
CODEGEN25_7B_MONO_MODEL_DIR = DEPENDENCIES_DIR / "codegen25-7b-mono"
CODELLAMA_7B_PYTHON_HF_MODEL_DIR = DEPENDENCIES_DIR / "CodeLlama-7b-Python-hf"
CODELLAMA_7B_HF_MODEL_DIR = DEPENDENCIES_DIR / "CodeLlama-7b-hf"

if __name__ == "__main__":
    print("[*] CLI runs on...")
    print(f"    Project: {PROJECT_DIR}")
    print(f"    Build: {BUILD_DIR}")
    print(f"    Dependencies: {DEPENDENCIES_DIR}")
