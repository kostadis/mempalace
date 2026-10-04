"""The MCP server is a package; the public import path is unchanged."""

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FRAGMENTS = (
    "_logging",
    "_guards",
    "_session",
    "tools_read",
    "tools_write",
    "tools_kg",
    "tools_diary",
    "tools_coord",
    "schemas",
    "protocol",
    "http",
    "runtime",
)


def test_mcp_server_is_a_package():
    import mempalace.mcp_server as mcp

    assert hasattr(mcp, "__path__")
    assert callable(mcp.main)
    assert callable(mcp.handle_request)
    assert "mempalace_status" in mcp.TOOLS
    assert callable(mcp.TOOLS["mempalace_status"]["handler"])


@pytest.mark.parametrize("name", FRAGMENTS)
def test_fragments_refuse_direct_import(name):
    with pytest.raises(ImportError, match="implementation fragment"):
        importlib.import_module(f"mempalace.mcp_server.{name}")


def test_python_m_mcp_server_still_runs():
    """``python -m mempalace.mcp_server`` must still be a valid entry."""
    proc = subprocess.run(
        [sys.executable, "-m", "mempalace.mcp_server"],
        input=b"",
        capture_output=True,
        timeout=60,
        cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", errors="replace")
    assert proc.stdout == b""


def test_fragment_files_exist():
    pkg = REPO_ROOT / "mempalace" / "mcp_server"
    missing = [name for name in FRAGMENTS if not (pkg / f"{name}.py").is_file()]
    assert missing == []
    assert not (REPO_ROOT / "mempalace" / "mcp_server.py").exists()
