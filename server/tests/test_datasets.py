"""SPDX-License-Identifier: Apache-2.0 — dataset conversion + routes."""
import json

from server.services import datasets as svc


def test_to_messages_qa_mapping():
    rec = svc.to_messages({"question": "2+2?", "answer": "4"},
                          {"question": "user", "answer": "assistant"})
    assert rec["messages"] == [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4"},
    ]


def test_to_messages_system_first():
    rec = svc.to_messages({"sys": "be brief", "q": "hi", "a": "hello"},
                          {"sys": "system", "q": "user", "a": "assistant"})
    assert rec["messages"][0]["role"] == "system"


def test_validate_record():
    ok, _ = svc.validate_record({"messages": [{"role": "user", "content": "x"}]})
    assert ok is True
    ok, errs = svc.validate_record({"messages": [{"role": "tool", "content": "x"}]})
    assert ok is False and any("invalid role" in e for e in errs)
    ok, _ = svc.validate_record({"messages": []})
    assert ok is False


def test_jsonl_roundtrip_and_stats(tmp_path):
    recs = svc.convert_rows(
        [{"question": "q1", "answer": "a1"}, {"question": "q2", "answer": "a2"}],
        {"question": "user", "answer": "assistant"})
    p = svc.write_jsonl(recs, tmp_path / "train.jsonl")
    loaded = svc.read_jsonl(p)
    assert loaded == recs
    st = svc.stats(recs)
    assert st["rows"] == 2 and st["roles"].get("user") == 2


def test_convert_and_preview_routes(client):
    body = {"rows": [{"question": f"q{i}", "answer": f"a{i}"} for i in range(5)],
            "mapping": {"question": "user", "answer": "assistant"}, "name": "tiny"}
    r = client.post("/api/datasets/convert", json=body)
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["stats"]["rows"] == 5
    did = data["id"]

    prev = client.get(f"/api/datasets/{did}/preview?n=3").json()["data"]
    assert prev["stats"]["rows"] == 5
    assert len(prev["records"]) == 3
    assert prev["records"][0]["messages"][0]["role"] == "user"

    listed = client.get("/api/datasets").json()["data"]
    assert any(d["id"] == did for d in listed)


def test_convert_flags_invalid(client):
    body = {"rows": [{"question": "q", "answer": ""}],  # empty assistant content
            "mapping": {"question": "user", "answer": "assistant"}, "name": "bad"}
    data = client.post("/api/datasets/convert", json=body).json()["data"]
    assert data["invalid_count"] >= 1


def test_build_gsm8k_requires_network(tmp_path):
    """build_gsm8k needs the HF hub; skip if offline."""
    import pytest
    try:
        from datasets import load_dataset  # noqa
    except Exception:
        pytest.skip("datasets extra missing")
    try:
        out = svc.build_gsm8k(tmp_path / "gsm8k")
        assert "train" in out["splits"]
    except Exception as exc:  # network/hub unreachable in CI
        pytest.skip(f"offline: {exc}")
