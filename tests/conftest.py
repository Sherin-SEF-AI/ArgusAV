import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(autouse=True)
def private_state(tmp_path, monkeypatch):
    """Audit log and the active-recording flag go to a temp dir, never the operator's
    (a test recording must not make the real rig's offline tools refuse to run)."""
    from avrig import util
    d = Path(tmp_path) / "state"
    monkeypatch.setattr(util, "STATE_DIR", d)
    monkeypatch.setattr(util, "AUDIT_FILE", d / "audit.log")
    monkeypatch.setattr(util, "RECORDING_FLAG", d / "recording.json")
    monkeypatch.setattr(util, "rigcap_capture", lambda: None)   # the operator's rigcap is not the test's
    from avrig import overview, study                            # caches and studies: never the operator's
    monkeypatch.setattr(overview, "DATA_DIR", str(d))
    monkeypatch.setattr(overview, "CACHE", str(d / "overview_cache.json"))
    monkeypatch.setattr(study, "DATA_DIR", str(d))
