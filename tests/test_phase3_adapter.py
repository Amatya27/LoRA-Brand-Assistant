"""Portable tests for Phase 3 adapter integration: no MLX, no weights, no network.

The load ledger is normally written by scripts/serve_mlx_with_load_ledger.py
inside the MLX server process. Here a fake transport writes the same events so
the client-side checks can be exercised without Apple Silicon.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

import brand_assistant_rag as rag
from brand_assistant_load_ledger import (
    compare_adapter_tensors,
    messages_sha256,
    verify_request_against_ledger,
)
from brand_assistant_openai_backend import OpenAICompatibleConfig
from brand_assistant_runs import RunOptions, run_pipeline

from test_phase3_inference import FIXTURE, FakeOpener, completion, no_retrieval, valid_output

MODEL = "mlx-community/gemma-3-1b-it-qat-4bit"
REPO = Path(__file__).resolve().parent.parent


# --- tensor comparison -----------------------------------------------------

def _cmp(file_tensors, params):
    return compare_adapter_tensors(file_tensors, params, equal=lambda a, b: a == b, shape=lambda a: (len(a),))


def test_tensor_comparison_requires_every_tensor_present_and_equal():
    params = {"l.0.q.lora_a": [1, 2], "l.0.q.lora_b": [3], "l.0.q.weight": [9, 9, 9]}
    assert _cmp({"l.0.q.lora_a": [1, 2], "l.0.q.lora_b": [3]}, params)["all_tensors_match"] is True
    assert _cmp({"l.0.q.lora_a": [1, 5], "l.0.q.lora_b": [3]}, params)["value_mismatch"] == ["l.0.q.lora_a"]
    assert _cmp({"l.0.q.lora_a": [1], "l.0.q.lora_b": [3]}, params)["shape_mismatch"] == ["l.0.q.lora_a"]
    assert _cmp({"l.0.x.lora_a": [1, 2], "l.0.q.lora_b": [3]}, params)["all_tensors_match"] is False
    # a LoRA parameter in the model that the file did not supply means a partial load
    assert _cmp({"l.0.q.lora_a": [1, 2]}, params)["lora_params_not_in_file"] == ["l.0.q.lora_b"]
    assert _cmp({}, params)["all_tensors_match"] is False


# --- ledger verification ---------------------------------------------------

def _events(digest, adapter="adapters/brand_assistant_best", sha="abc", in_flight=1, served_adapter=None, t=100.0):
    served = adapter if served_adapter is None else served_adapter
    load = {"event": "model_loaded", "pid": 7, "time": t - 1, "model_key": [MODEL, served, None]}
    if served:
        load["adapter_inspection"] = {
            "adapters_safetensors_sha256": sha,
            "lora_module_count": 112,
            "tensor_comparison": {"all_tensors_match": True},
        }
    return [
        {"event": "server_started", "pid": 7, "time": 1.0},
        load,
        {"event": "request_started", "pid": 7, "time": t, "messages_sha256": digest,
         "requested_model": MODEL, "requested_adapter": adapter, "model_key": [MODEL, served, None],
         "in_flight": in_flight},
    ]


def _verify(events, adapter="adapters/brand_assistant_best", sha="abc", not_before=50.0):
    return verify_request_against_ledger(
        events, messages_digest="d", requested_model=MODEL, requested_adapter=adapter,
        expected_adapter_sha256=sha, not_before=not_before,
    )


def test_ledger_verifies_adapter_only_when_every_check_passes():
    assert _verify(_events("d"))["verified"] is True
    assert _verify(_events("d", sha="other"))["checks"]["adapter_file_hash_matches_expected"] is False
    assert _verify(_events("d", in_flight=2))["checks"]["single_request_in_flight"] is False
    assert _verify(_events("other"))["checks"] == {"request_found_in_ledger": False}
    assert _verify(_events("d"), not_before=500.0)["verified"] is False  # stale event from an earlier request


def test_ledger_detects_request_served_without_the_adapter():
    events = _events("d", served_adapter="")
    events[-1]["model_key"] = [MODEL, None, None]
    events[1] = {"event": "model_loaded", "pid": 7, "time": 99.0, "model_key": [MODEL, None, None]}
    result = _verify(events)
    assert result["verified"] is False
    assert result["checks"]["served_adapter_matches_request"] is False


def test_ledger_confirms_base_request_had_no_adapter():
    events = _events("d", adapter=None, served_adapter="")
    events[-1]["model_key"] = [MODEL, None, None]
    events[1] = {"event": "model_loaded", "pid": 7, "time": 99.0, "model_key": [MODEL, None, None]}
    result = _verify(events, adapter=None, sha=None)
    assert result["verified"] is True and result["checks"]["no_adapter_loaded"] is True


# --- run pipeline with a ledger-writing fake server ------------------------

class LedgerServer(FakeOpener):
    """Fake transport that also writes the ledger events the instrumented server would."""

    def __init__(self, ledger: Path, sha: str | None, *behaviours, loaded_adapter="__requested__"):
        super().__init__(*behaviours)
        self.ledger, self.sha, self.loaded_adapter = ledger, sha, loaded_adapter

    def __call__(self, request, timeout=None):
        body = json.loads(request.data.decode())
        adapter = body.get("adapters") if self.loaded_adapter == "__requested__" else self.loaded_adapter
        now = time.time()
        load = {"event": "model_loaded", "pid": 1, "time": now, "model_key": [body["model"], adapter, None]}
        if adapter:
            load["adapter_inspection"] = {
                "adapters_safetensors_sha256": self.sha,
                "lora_module_count": 112,
                "tensor_comparison": {"all_tensors_match": True},
            }
        req = {"event": "request_started", "pid": 1, "time": now, "messages_sha256": messages_sha256(body["messages"]),
               "requested_model": body["model"], "requested_adapter": body.get("adapters"),
               "model_key": [body["model"], adapter, None], "in_flight": 1}
        with open(self.ledger, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(load) + "\n" + json.dumps(req) + "\n")
        return super().__call__(request, timeout)


def _adapter_dir(tmp_path, model=MODEL) -> Path:
    path = tmp_path / "adapter"
    path.mkdir()
    (path / "adapters.safetensors").write_bytes(b"synthetic adapter bytes")
    (path / "adapter_config.json").write_text(json.dumps({
        "model": model, "fine_tune_type": "lora", "num_layers": 16,
        "lora_parameters": {"rank": 8, "dropout": 0.0, "scale": 20.0},
        "adapter_path": "adapters/brand_assistant_1b_qat", "iters": 600,
    }))
    return path


def _sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256((path / "adapters.safetensors").read_bytes()).hexdigest()


def _adapter_options(tmp_path, adapter: Path, ledger: Path | None) -> RunOptions:
    return RunOptions(
        mode="adapter-generation", packet_input=FIXTURE, runs_root=tmp_path / "runs",
        openai_config=OpenAICompatibleConfig(adapter_path=str(adapter)), adapter_path_is_local=True,
        server_load_ledger=ledger,
    )


def test_adapter_load_verified_with_matching_runtime_ledger(tmp_path):
    adapter, ledger = _adapter_dir(tmp_path), tmp_path / "ledger.jsonl"
    server = LedgerServer(ledger, _sha(adapter), completion(json.dumps(valid_output())))
    record = run_pipeline(_adapter_options(tmp_path, adapter, ledger), retrieve=no_retrieval, opener=server)
    info = record["generation"]["adapter"]
    assert info["adapter_load_verified"] is True
    assert record["generation"]["runtime_verification"]["method"] == "server_load_ledger"
    assert info["local_validation"]["adapter_config"]["training_output_path_in_config"] == "adapters/brand_assistant_1b_qat"
    assert record["status"]["send_ready"] is False


def test_adapter_not_verified_without_ledger_even_on_http_200(tmp_path):
    adapter = _adapter_dir(tmp_path)
    record = run_pipeline(
        _adapter_options(tmp_path, adapter, None), retrieve=no_retrieval,
        opener=FakeOpener(completion(json.dumps(valid_output()))),
    )
    info = record["generation"]["adapter"]
    assert record["generation"]["http_status"] == 200
    assert info["adapter_load_verified"] is False
    assert "HTTP status 200" in info["adapter_load_verification_note"]


def test_adapter_not_verified_when_server_loaded_different_file(tmp_path):
    adapter, ledger = _adapter_dir(tmp_path), tmp_path / "ledger.jsonl"
    server = LedgerServer(ledger, "0" * 64, completion(json.dumps(valid_output())))
    record = run_pipeline(_adapter_options(tmp_path, adapter, ledger), retrieve=no_retrieval, opener=server)
    assert record["generation"]["adapter"]["adapter_load_verified"] is False
    assert record["generation"]["runtime_verification"]["checks"]["adapter_file_hash_matches_expected"] is False


def test_adapter_not_verified_when_server_served_base_model(tmp_path):
    adapter, ledger = _adapter_dir(tmp_path), tmp_path / "ledger.jsonl"
    server = LedgerServer(ledger, _sha(adapter), completion(json.dumps(valid_output())), loaded_adapter=None)
    record = run_pipeline(_adapter_options(tmp_path, adapter, ledger), retrieve=no_retrieval, opener=server)
    assert record["generation"]["adapter"]["adapter_load_verified"] is False
    assert record["generation"]["runtime_verification"]["checks"]["served_adapter_matches_request"] is False


def test_base_run_records_runtime_no_adapter_check(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    server = LedgerServer(ledger, None, completion(json.dumps(valid_output())))
    record = run_pipeline(
        RunOptions(mode="fixed-generation", packet_input=FIXTURE, runs_root=tmp_path / "runs", server_load_ledger=ledger),
        retrieve=no_retrieval, opener=server,
    )
    assert "adapters" not in server.requests[0]["body"]
    assert record["generation"]["base_model_runtime_check"]["no_adapter_loaded_verified"] is True


def test_adapter_for_a_different_base_model_is_refused_before_any_request(tmp_path):
    adapter = _adapter_dir(tmp_path, model="mlx-community/some-other-model")
    opener = FakeOpener()
    record = run_pipeline(_adapter_options(tmp_path, adapter, None), retrieve=no_retrieval, opener=opener)
    assert opener.requests == []
    assert record["status"]["execution_status"] == "failed"
    assert record["errors"][0]["type"] == "adapter_incompatible"


def test_base_and_adapter_replays_share_packet_and_prompt_hashes(tmp_path):
    adapter = _adapter_dir(tmp_path)
    base = run_pipeline(
        RunOptions(mode="fixed-generation", packet_input=FIXTURE, runs_root=tmp_path / "runs"),
        retrieve=no_retrieval, opener=FakeOpener(completion(json.dumps(valid_output()))),
    )
    tuned = run_pipeline(
        _adapter_options(tmp_path, adapter, None), retrieve=no_retrieval,
        opener=FakeOpener(completion(json.dumps(valid_output()))),
    )
    assert base["evidence"]["packet_input_sha256"] == tuned["evidence"]["packet_input_sha256"]
    assert base["generation"]["prompt_sha256"] == tuned["generation"]["prompt_sha256"]
    keep = ("model", "temperature", "max_tokens")
    assert {k: base["generation"]["settings"][k] for k in keep} == {k: tuned["generation"]["settings"][k] for k in keep}
    assert base["generation"]["settings"]["adapter_path"] is None and tuned["generation"]["settings"]["adapter_path"]


# --- distinct incomplete outcomes ------------------------------------------

def test_unknown_citation_ids_have_their_own_incomplete_reason(tmp_path):
    out = valid_output(citations=("[1]", "[9]"))
    record = run_pipeline(
        RunOptions(mode="fixed-generation", packet_input=FIXTURE, runs_root=tmp_path / "runs"),
        retrieve=no_retrieval, opener=FakeOpener(completion(json.dumps(out))),
    )
    assert record["status"]["incomplete_reason"] == "unknown_citation_ids"


def test_unescaped_quotes_are_malformed_json_not_repaired(tmp_path):
    raw = json.dumps(valid_output()).replace("fictional trail-gear brand", 'brand with a "Just Run" slogan')
    record = run_pipeline(
        RunOptions(mode="fixed-generation", packet_input=FIXTURE, runs_root=tmp_path / "runs"),
        retrieve=no_retrieval, opener=FakeOpener(completion(raw)),
    )
    assert record["status"]["incomplete_reason"] == "malformed_json"
    assert (Path(record["run_dir"]) / "raw_output.txt").read_text() == raw
    assert not (Path(record["run_dir"]) / "parsed_output.json").exists()


# --- search diagnostics ----------------------------------------------------

class _FlakyClient:
    def __init__(self):
        self.calls = 0

    def text(self, query, **kw):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("No results found.")
        return [{"href": "https://example.com/a", "title": "A", "body": "b"}]


def test_search_errors_are_recorded_not_silently_dropped():
    log: list = []
    token = rag._SEARCH_LOG.set(log)
    try:
        client = _FlakyClient()
        assert rag._search_with_client(client, "q1") == []
        assert len(rag._search_with_client(client, "q2")) == 1
    finally:
        rag._SEARCH_LOG.reset(token)
    summary = rag.summarize_search_log(log)
    assert summary["search_calls"] == 2 and summary["search_calls_raised"] == 1
    assert summary["errors"][0]["query"] == "q1"


def test_research_only_with_failed_search_is_insufficient_evidence_not_absence(tmp_path):
    def failed_retrieve(company, person, **kw):
        return {"company": company, "person": person, "sources": [], "evidence_passages": [],
                "grounded_prompt": rag.build_prompt(company, person, []), "profiles": {},
                "search_diagnostics": {"search_calls": 3, "search_calls_raised": 3, "errors": []}}
    record = run_pipeline(
        RunOptions(mode="research-only", company="Synthetic Quiet Co", runs_root=tmp_path / "runs"),
        retrieve=failed_retrieve,
    )
    assert record["status"]["incomplete_reason"] == "insufficient_evidence"
    assert record["evidence"]["search_diagnostics"]["search_calls_raised"] == 3
    assert "not a finding" in record["evidence"]["insufficient_evidence_note"]


# --- delivered checkpoint and docs -----------------------------------------

def test_docs_use_delivered_adapter_directory():
    """Inference examples must use the delivered checkpoint; the training command may keep its output path."""
    import re
    for doc in ("README.md", "docs/phase3_inference.md"):
        text = (REPO / doc).read_text(encoding="utf-8")
        assert "adapters/brand_assistant_best" in text, doc
        blocks = re.findall(r"```bash\n(.*?)```", text, flags=re.S)
        inference = [b for b in blocks if "run_brand_assistant.py" in b and "--adapter-path" in b]
        assert inference, doc
        for block in inference:
            assert "brand_assistant_1b_qat" not in block, (doc, block)
            assert "adapters/brand_assistant_best" in block, (doc, block)


@pytest.mark.skipif(not (REPO / "adapters/brand_assistant_best/adapters.safetensors").exists(),
                    reason="Phase 2 adapter not checked out")
def test_delivered_adapter_matches_published_hash_and_base_model():
    path = REPO / "adapters/brand_assistant_best"
    assert _sha(path) == "4fbc4b787f6eb585b3540dbe1f242d818f77f6953199b2910f4dd9a2eb57af52"
    assert json.loads((path / "adapter_config.json").read_text())["model"] == MODEL
