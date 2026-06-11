from pathlib import Path


BUNDLED_FRONTEND_PATH = Path(__file__).parent / "frontend"


def is_frontend_path(path: Path | None) -> bool:
    return bool(path and path.is_dir() and (path / "index.html").is_file())


def get_bundled_frontend_path() -> Path | None:
    if is_frontend_path(BUNDLED_FRONTEND_PATH):
        return BUNDLED_FRONTEND_PATH
    return None
