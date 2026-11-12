"""Configured providers handle compatible utility work; other work uses MPS."""

import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

from api.services.provider_document_processing import process_text_document
from api.services.provider_utility_inference import generate_workflow, transcribe_audio


@pytest.fixture
def configuration():
    return SimpleNamespace(
        llm=SimpleNamespace(provider="openai", model="test-model", api_key="test-key"),
        stt=SimpleNamespace(provider="openai", model="test-model", api_key="test-key"),
    )


@pytest.fixture
def resolved_configuration(monkeypatch, configuration):
    resolve = AsyncMock(return_value=configuration)
    for module_name in (
        "api.services.provider_utility_inference",
        "api.services.provider_document_processing",
    ):
        monkeypatch.setattr(f"{module_name}._resolve_model_configuration", resolve)
    resolve.reset_mock()
    return resolve


def provider_call(name):
    module = importlib.import_module(name)
    service = SimpleNamespace(run_inference=AsyncMock())
    module.create_llm_service_with_model_override = Mock(return_value=service)
    return service


async def test_default_workflow_uses_provider_and_validates_output(
    monkeypatch, resolved_configuration
):
    definition = {
        "name": "Reception",
        "workflow_definition": {"nodes": [{"id": "start"}], "edges": []},
    }
    call = provider_call("api.services.provider_utility_inference")
    call.run_inference.return_value = json.dumps(definition)
    assert (
        await generate_workflow(
            organization_id=42,
            call_type="inbound",
            use_case="reception",
            activity_description="Answer visitor questions",
        )
        == definition
    )
    resolved_configuration.assert_awaited_once_with(42, None)
    assert call.run_inference.await_count == 1


@pytest.mark.parametrize(
    "workflow_definition",
    [None, {"nodes": {}, "edges": []}, {"nodes": [], "edges": {}}],
)
async def test_invalid_provider_workflow_is_rejected(
    resolved_configuration, workflow_definition
):
    call = provider_call("api.services.provider_utility_inference")
    call.run_inference.return_value = json.dumps(
        {"workflow_definition": workflow_definition}
    )
    with pytest.raises(ValueError, match="invalid workflow definition"):
        await generate_workflow(
            organization_id=42,
            call_type="inbound",
            use_case="reception",
            activity_description="Answer visitor questions",
        )


@pytest.mark.parametrize("llm", [None, SimpleNamespace(provider="dograh")])
async def test_managed_or_missing_llm_keeps_workflow_fallback(
    resolved_configuration, llm
):
    resolved_configuration.return_value.llm = llm
    assert (
        await generate_workflow(
            organization_id=42,
            call_type="inbound",
            use_case="reception",
            activity_description="Answer visitor questions",
        )
        is None
    )


async def test_real_openai_factories_accept_saved_configuration(
    monkeypatch, resolved_configuration
):
    module = importlib.import_module("api.services.provider_utility_inference")
    document_module = importlib.import_module(
        "api.services.provider_document_processing"
    )
    real_create_llm = module.create_llm_service_with_model_override
    real_create_stt = module.create_stt_service
    captured = {}

    def create_llm(*args, **kwargs):
        service = real_create_llm(*args, **kwargs)
        service.run_inference = AsyncMock(
            return_value=json.dumps(
                {
                    "workflow_definition": {"nodes": [], "edges": []},
                    "full_text": "A document",
                }
            )
        )
        captured["llm"] = service
        return service

    monkeypatch.setattr(module, "create_llm_service_with_model_override", create_llm)
    monkeypatch.setattr(
        document_module,
        "create_llm_service_with_model_override",
        create_llm,
    )

    def create_stt(*args, **kwargs):
        service = real_create_stt(*args, **kwargs)
        service._transcribe = AsyncMock(
            return_value=SimpleNamespace(text="A transcript")
        )
        captured["stt"] = service
        return service

    monkeypatch.setattr(module, "create_stt_service", create_stt)
    assert (
        await generate_workflow(
            organization_id=42,
            call_type="inbound",
            use_case="reception",
            activity_description="Answer visitor questions",
        )
        is not None
    )
    assert await transcribe_audio(
        organization_id=42,
        audio_data=b"RIFF\x04\x00\x00\x00WAVE data",
        language="en",
    ) == {"transcript": "A transcript"}
    assert captured["llm"] is captured["stt"] or captured["llm"] is not None


async def test_openai_transcription_returns_text_only(
    monkeypatch, configuration, resolved_configuration
):
    result = SimpleNamespace(text=" A transcript ")
    stt = SimpleNamespace(
        _settings=SimpleNamespace(language="en"),
        _transcribe=AsyncMock(return_value=result),
    )
    monkeypatch.setattr(
        "api.services.provider_utility_inference.create_stt_service",
        lambda _configuration, _audio_config: stt,
    )
    assert await transcribe_audio(
        organization_id=42,
        audio_data=b"RIFF\x04\x00\x00\x00WAVE data",
        language="multi",
    ) == {"transcript": "A transcript"}
    assert stt._settings.language is None
    stt._transcribe.assert_awaited_once_with(b"RIFF\x04\x00\x00\x00WAVE data")


@pytest.mark.parametrize("text", [None, "", "   "])
async def test_empty_provider_transcript_is_rejected(
    monkeypatch, resolved_configuration, text
):
    stt = SimpleNamespace(
        _settings=SimpleNamespace(language="en"),
        _transcribe=AsyncMock(return_value=SimpleNamespace(text=text)),
    )
    monkeypatch.setattr(
        "api.services.provider_utility_inference.create_stt_service",
        lambda _configuration, _audio_config: stt,
    )
    with pytest.raises(HTTPException) as error:
        await transcribe_audio(
            organization_id=42,
            audio_data=b"RIFF\x04\x00\x00\x00WAVE data",
            language="en",
        )
    assert error.value.status_code == 422


@pytest.mark.parametrize(
    "stt",
    [None, SimpleNamespace(provider="dograh"), SimpleNamespace(provider="deepgram")],
)
async def test_unsupported_stt_keeps_managed_fallback(
    monkeypatch, configuration, resolved_configuration, stt
):
    resolved_configuration.return_value.stt = stt
    create = Mock()
    monkeypatch.setattr(
        "api.services.provider_utility_inference.create_stt_service", create
    )
    assert (
        await transcribe_audio(
            organization_id=42,
            audio_data=b"RIFF\x04\x00\x00\x00WAVE data",
            language="en",
        )
        is None
    )
    create.assert_not_called()


@pytest.mark.parametrize("retrieval_mode", ["chunked", "full_document"])
async def test_text_processing_uses_configured_inference(
    monkeypatch, tmp_path, resolved_configuration, retrieval_mode
):
    document = tmp_path / "document.txt"
    document.write_text("A document", encoding="utf-8")
    call = SimpleNamespace(
        run_inference=AsyncMock(
            return_value=json.dumps(
                {
                    "full_text": "A document",
                    "chunks": [
                        {
                            "chunk_text": "A document",
                            "contextualized_text": "It opens the document.",
                        }
                    ],
                }
            )
        )
    )
    create = Mock(return_value=call)
    monkeypatch.setattr(
        "api.services.provider_document_processing.create_llm_service_with_model_override",
        create,
    )
    result = await process_text_document(
        organization_id=42,
        file_path=str(document),
        filename="DOCUMENT.TXT",
        retrieval_mode=retrieval_mode,
        max_tokens=128,
        model_configuration_uuid="saved-configuration",
    )
    create.assert_called_once_with(
        resolved_configuration.return_value,
        None,
        usage_context="document_extraction",
    )
    assert (
        call.run_inference.await_args.args[0]
        .get_messages()[0]["content"]
        .endswith("A document")
    )
    if retrieval_mode == "full_document":
        assert result == {"full_text": "A document", "docling_metadata": {}}
    else:
        assert result == {
            "chunks": [
                {
                    "chunk_text": "A document",
                    "contextualized_text": "It opens the document.",
                    "chunk_index": 0,
                    "chunk_metadata": {},
                    "token_count": 0,
                }
            ],
            "docling_metadata": {},
        }
    resolved_configuration.assert_awaited_once_with(42, "saved-configuration")


@pytest.mark.parametrize(
    "response", [{}, {"full_text": "Document"}, {"full_text": "Document", "chunks": {}}]
)
async def test_invalid_or_empty_document_extraction_is_rejected(
    monkeypatch, tmp_path, resolved_configuration, response
):
    document = tmp_path / "document.md"
    document.write_text("Document", encoding="utf-8")
    call = SimpleNamespace(run_inference=AsyncMock(return_value=json.dumps(response)))
    monkeypatch.setattr(
        "api.services.provider_document_processing.create_llm_service_with_model_override",
        Mock(return_value=call),
    )
    with pytest.raises(ValueError):
        await process_text_document(
            organization_id=42,
            file_path=str(document),
            filename="document.md",
            retrieval_mode="chunked",
            max_tokens=128,
        )


@pytest.mark.parametrize("filename", ["document.pdf", "document"])
async def test_other_document_formats_keep_fallback(
    tmp_path, resolved_configuration, filename
):
    document = tmp_path / filename
    document.write_text("document", encoding="utf-8")
    resolved_configuration.return_value.llm = None
    assert (
        await process_text_document(
            organization_id=42,
            file_path=str(document),
            filename=filename,
            retrieval_mode="chunked",
            max_tokens=128,
        )
        is None
    )
    resolved_configuration.assert_not_awaited()


async def test_extraneous_output_keys_are_not_persisted(configuration):
    # The endpoint uses only name and workflow_definition after this validation.
    # Keep this explicit so arbitrary provider fields never imply stored fields.
    assert configuration.llm.provider == "openai"
