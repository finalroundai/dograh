"""Existing utility endpoints pass provider work through their normal contracts."""

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

from api.routes.knowledge_base import process_document
from api.routes.workflow import (
    CreateWorkflowTemplateRequest,
    create_workflow_from_template,
)
from api.routes.workflow_recording import transcribe_audio
from api.schemas.knowledge_base import ProcessDocumentRequestSchema
from api.services.knowledge_base_content import enqueue_document_processing


def user():
    return SimpleNamespace(id=7, provider_id="provider-7", selected_organization_id=42)


async def test_recording_transcription_passes_configuration_and_keeps_response(
    monkeypatch,
):
    configured = AsyncMock(return_value={"transcript": "A transcript"})
    mps = AsyncMock()
    monkeypatch.setattr(
        "api.routes.workflow_recording.configured_transcribe_audio", configured
    )
    monkeypatch.setattr(
        "api.routes.workflow_recording.mps_service_key_client.transcribe_audio", mps
    )
    upload = SimpleNamespace(
        read=AsyncMock(return_value=b"RIFF\x04\x00\x00\x00WAVE data"),
        filename="recording.wav",
        content_type="audio/wav",
    )
    result = await transcribe_audio(
        upload,
        language="en",
        model_configuration_uuid="saved-configuration",
        user=user(),
    )
    assert result == {"transcript": "A transcript"}
    configured.assert_awaited_once_with(
        organization_id=42,
        audio_data=upload.read.return_value,
        language="en",
        model_configuration_uuid="saved-configuration",
    )
    mps.assert_not_awaited()


async def test_provider_workflow_returns_the_created_workflow(monkeypatch):
    definition = {
        "name": "Provider workflow",
        "workflow_definition": {"nodes": [], "edges": []},
    }
    generated = AsyncMock(return_value=definition)
    created = SimpleNamespace(
        id=17,
        workflow_uuid="workflow-17",
        name=definition["name"],
        status="published",
        created_at=datetime.now(timezone.utc),
        current_definition_id=3,
        template_context_variables=[],
        call_disposition_codes=[],
        workflow_configurations=[],
    )
    create = AsyncMock(return_value=created)
    monkeypatch.setattr("api.routes.workflow.generate_workflow", generated)
    monkeypatch.setattr("api.routes.workflow.db_client.create_workflow", create)
    monkeypatch.setattr("api.routes.workflow.capture_tenant_event", AsyncMock())
    result = await create_workflow_from_template(
        CreateWorkflowTemplateRequest(
            call_type="inbound",
            use_case="reception",
            activity_description="Answer visitor questions",
            model_configuration_uuid="saved-configuration",
        ),
        user(),
    )
    assert result["id"] == 17
    assert result["workflow_definition"] == definition["workflow_definition"]
    assert (
        create.await_args.kwargs["workflow_definition"]
        == definition["workflow_definition"]
    )
    generated.assert_awaited_once_with(
        organization_id=42,
        call_type="inbound",
        use_case="reception",
        activity_description="Answer visitor questions",
        model_configuration_uuid="saved-configuration",
    )


async def test_document_endpoint_enqueues_provider_configuration(monkeypatch):
    now = datetime.now(timezone.utc)
    created = SimpleNamespace(id=17, created_at=now, updated_at=now)
    create = AsyncMock(return_value=created)
    enqueue = AsyncMock()
    monkeypatch.setattr("api.routes.knowledge_base.db_client.create_document", create)
    monkeypatch.setattr(
        "api.routes.knowledge_base.enqueue_document_processing", enqueue
    )
    monkeypatch.setattr("api.routes.knowledge_base.capture_tenant_event", AsyncMock())
    request = ProcessDocumentRequestSchema(
        document_uuid="document-uuid",
        s3_key="knowledge_base/42/document-uuid/document.txt",
        model_configuration_uuid="saved-configuration",
    )
    result = await process_document(request, user())
    assert result.document_uuid == "document-uuid"
    assert result.processing_status == "pending"
    enqueue.assert_awaited_once_with(
        document_id=17,
        s3_key=request.s3_key,
        organization_id=42,
        created_by_provider_id="provider-7",
        retrieval_mode="chunked",
        model_configuration_uuid="saved-configuration",
    )


async def test_enqueue_document_processing_forwards_configuration(monkeypatch):
    enqueue = AsyncMock()
    monkeypatch.setattr("api.services.knowledge_base_content.enqueue_job", enqueue)
    await enqueue_document_processing(
        document_id=7,
        s3_key="knowledge_base/42/document-uuid/document.txt",
        organization_id=42,
        created_by_provider_id="provider-7",
        retrieval_mode="chunked",
        model_configuration_uuid="saved-configuration",
    )
    from api.tasks.function_names import FunctionNames

    enqueue.assert_awaited_once_with(
        FunctionNames.PROCESS_KNOWLEDGE_BASE_DOCUMENT,
        7,
        "knowledge_base/42/document-uuid/document.txt",
        42,
        "provider-7",
        128,
        "chunked",
        "saved-configuration",
    )
