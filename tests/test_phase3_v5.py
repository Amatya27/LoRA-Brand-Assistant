"""Portable tests for the v5 compatibility changes: prompt v3.3 and 8-layer adapters.

No MLX, weights or network. Fixtures are SYNTHETIC (see tests/test_phase3_v3.py).
"""

from __future__ import annotations

import json

from brand_assistant_load_ledger import compare_adapter_tensors
from brand_assistant_runs import run_pipeline
from brand_assistant_v3 import PROMPT_V32, PROMPT_V33, V3_PROMPT_VERSIONS, build_v3_prompt, teacher_rules

from test_phase3_inference import FakeOpener, completion, no_retrieval
from test_phase3_v3 import _eval, good_output, packet, v3_options, write_packet


def test_v33_adds_instagram_vendor_and_unknowns_rules_only():
    v32, v33 = teacher_rules(PROMPT_V32), teacher_rules(PROMPT_V33)
    assert PROMPT_V33 in V3_PROMPT_VERSIONS
    assert "Instagram only (no TikTok, YouTube, podcasts or other channels)" in v33
    assert '"According to AdLibrary, ...", "Attentive reports that ..."' in v33
    assert '"unknowns": ["one specific open question, e.g. Whether the company works with creators"]' in v33
    assert '"unknowns": ["what the evidence does not show"]' in v32 and '"unknowns": ["what the evidence does not show"]' not in v33
    # everything else is shared: only the email rule (extended on the same line) and the unknowns example differ
    missing = [line for line in v32.splitlines() if line not in v33.splitlines()]
    assert missing == [
        "- The email is from our team to the company, proposing one collaborative Instagram post.",
        '  "unknowns": ["what the evidence does not show"],',
    ]
    assert len(v33.splitlines()) == len(v32.splitlines()) + 5


def test_v33_prompt_renders_packet_unchanged_and_runs(tmp_path):
    pk = packet()
    prompt = build_v3_prompt(pk, PROMPT_V33)
    assert prompt.endswith(build_v3_prompt(pk, PROMPT_V32).split("\n\nCompany: ", 1)[1])
    opener = FakeOpener(completion(json.dumps(good_output())))
    record = run_pipeline(v3_options(tmp_path, write_packet(tmp_path, pk), version=PROMPT_V33),
                          retrieve=no_retrieval, opener=opener)
    assert opener.requests[0]["body"]["messages"][0]["content"] == prompt
    assert record["generation"]["prompt_version"] == PROMPT_V33


def test_v33_example_unknown_copied_verbatim_is_a_placeholder():
    out = good_output()
    out["unknowns"] = ["One specific open question, e.g. Whether the company works with creators"]
    assert any("placeholders copied" in e for e in _eval(out)["v3"]["form_gate_errors"])


def test_tensor_check_accepts_an_adapter_on_a_subset_of_layers():
    """An 8-of-26-layer adapter: unadapted layers have no LoRA parameters, so nothing is 'missing'."""
    params = {f"model.layers.{i}.q.weight": [0] for i in range(26)}
    file_tensors = {}
    for i in range(18, 26):
        for kind in ("lora_a", "lora_b"):
            params[f"model.layers.{i}.q.{kind}"] = [i]
            file_tensors[f"model.layers.{i}.q.{kind}"] = [i]
    result = compare_adapter_tensors(file_tensors, params, equal=lambda a, b: a == b, shape=lambda a: (len(a),))
    assert result["all_tensors_match"] is True
    assert result["file_tensor_count"] == result["model_lora_param_count"] == 16
    # a model that also carries LoRA on layer 10 (e.g. a stale 16-layer load) is not a match
    params["model.layers.10.q.lora_a"] = [1]
    assert compare_adapter_tensors(file_tensors, params, equal=lambda a, b: a == b,
                                   shape=lambda a: (len(a),))["lora_params_not_in_file"] == ["model.layers.10.q.lora_a"]


def test_default_payload_has_no_penalty_fields_and_no_experiment_label(tmp_path):
    opener = FakeOpener(completion(json.dumps(good_output())))
    record = run_pipeline(v3_options(tmp_path, write_packet(tmp_path, packet()), version=PROMPT_V33),
                          retrieve=no_retrieval, opener=opener)
    body = opener.requests[0]["body"]
    assert "repetition_penalty" not in body and "repetition_context_size" not in body
    assert "decoding_experiment" not in record["generation"]


def test_repetition_penalty_is_sent_and_labelled(tmp_path):
    from brand_assistant_openai_backend import OpenAICompatibleConfig

    opener = FakeOpener(completion(json.dumps(good_output())))
    options = v3_options(tmp_path, write_packet(tmp_path, packet()), version=PROMPT_V33,
                         openai_config=OpenAICompatibleConfig(repetition_penalty=1.1, repetition_context_size=256))
    record = run_pipeline(options, retrieve=no_retrieval, opener=opener)
    body = opener.requests[0]["body"]
    assert body["repetition_penalty"] == 1.1 and body["repetition_context_size"] == 256
    assert record["generation"]["decoding_experiment"]["label"].startswith("DECODING DIAGNOSTIC")
    assert record["generation"]["settings"]["repetition_penalty"] == 1.1
