import pytest

from backend.grounding.builder import (
    MAX_FACTS,
    MAX_LIST_ITEMS,
    GroundedInstructionBuilder,
    _INSTRUCTION_SCHEMA,
    bounded_instruction_schema,
)
from backend.grounding.evidence import build_evidence_packet
from backend.providers.base import MalformedProviderResponse

from backend.tests.unit.test_grounding_instruction import _items
from backend.tests.unit.test_ollama_native_adapter import FORCE, TOOL, Recorder, _adapter, _events, _reply, _request


def _schema_of(builder):
    packet = build_evidence_packet("Genera una imagen del oppidum", _items())
    [tool] = builder.build_request(packet).tools
    return tool.function.parameters


def test_bounded_schema_limits_every_list_and_text():
    schema = bounded_instruction_schema()
    props = schema["properties"]
    facts = props["documented_facts"]
    assert facts["maxItems"] == MAX_FACTS
    for field in ("text", "text_en"):
        assert facts["items"]["properties"][field]["maxLength"] > 0
    assert facts["items"]["properties"]["reference_ids"]["maxItems"] > 0
    for name in ("inferred_elements", "uncertainties", "forbidden_additions"):
        assert props[name]["maxItems"] == MAX_LIST_ITEMS
        assert props[name]["items"]["maxLength"] > 0
    assert props["style"]["maxLength"] > 0 and props["view"]["maxLength"] > 0
    # The shared schema is left untouched.
    assert "maxItems" not in _INSTRUCTION_SCHEMA["properties"]["documented_facts"]


def test_builder_uses_bounds_only_when_requested():
    assert _schema_of(GroundedInstructionBuilder(object(), model="m")) == _INSTRUCTION_SCHEMA
    bounded = GroundedInstructionBuilder(object(), model="m", bounded_output=True)
    assert _schema_of(bounded) == bounded_instruction_schema()


@pytest.mark.asyncio
async def test_truncated_structured_output_reports_the_token_cap():
    adapter = _adapter(Recorder(_reply('{"documented_facts": [{"text": "mur', done_reason="length")))
    with pytest.raises(MalformedProviderResponse, match="OLLAMA_MAX_OUTPUT_TOKENS"):
        await _events(adapter, _request(tools=[TOOL], tool_choice=FORCE))
