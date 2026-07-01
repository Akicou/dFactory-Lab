"""SPDX-License-Identifier: Apache-2.0"""
import time

from server import db as dbmod
from server.jobs import JobRegistry


def _wait(reg, jid, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = reg.get(jid)
        if j and j.state in ("done", "error", "cancelled"):
            return j
        time.sleep(0.02)
    raise AssertionError(f"job {jid} did not finish")


def test_submit_completes(tmp_path):
    dbmod.init_db(tmp_path / "t.sqlite")
    reg = JobRegistry(tmp_path / "t.sqlite")

    def fn(job, update):
        update(0.5, "halfway")
        update.done(result={"merged": True})

    jid = reg.submit("merge", fn, payload={"a": 1})
    j = _wait(reg, jid)
    assert j.state == "done"
    assert j.progress == 1.0
    assert j.result == {"merged": True}


def test_cancel(tmp_path):
    dbmod.init_db(tmp_path / "t.sqlite")
    reg = JobRegistry(tmp_path / "t.sqlite")

    def fn(job, update):
        while not reg.is_cancelled(job.id):
            update(0.1, "working")
            time.sleep(0.01)
        update(0.1, "cancelled by caller")

    jid = reg.submit("train", fn)
    time.sleep(0.05)
    assert reg.cancel(jid) is True
    j = _wait(reg, jid)
    assert j.state == "cancelled"


def test_error_captured(tmp_path):
    dbmod.init_db(tmp_path / "t.sqlite")
    reg = JobRegistry(tmp_path / "t.sqlite")

    def fn(job, update):
        raise RuntimeError("boom")

    jid = reg.submit("export", fn)
    j = _wait(reg, jid)
    assert j.state == "error"
    assert "boom" in (j.error or "")


def test_reconcile_on_startup(tmp_path):
    p = tmp_path / "t.sqlite"
    dbmod.init_db(p)
    reg = JobRegistry(p)
    jid = reg.submit("download", lambda job, update: None)
    j = reg.get(jid)
    j.state = "running"  # simulate a crash mid-run
    reg._persist(j)
    reg2 = JobRegistry(p)  # new process
    n = reg2.reconcile_on_startup()
    assert n == 1
    assert reg2.get(jid).state == "error"
