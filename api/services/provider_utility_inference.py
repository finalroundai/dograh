"""Provider-backed workflow generation and uploaded-file transcription.

These are the two existing provider capabilities that can be invoked outside a
live voice pipeline. Document conversion is intentionally not handled here.
"""

from typing import Literal

from api.services.configuration.ai_model_configuration import (
    get_resolved_ai_model_configuration,
)
from api.services.configuration.model_connections import resolve_model_configuration
from api.services.configuration.registry import ServiceProviders
from api.services.gen_ai.json_parser import parse_llm_json
from api.services.pipecat.audio_config import AudioConfig
from api.services.pipecat.service_factory import (
    create_llm_service_with_model_override,
    create_stt_service,
)
from fastapi import HTTPException
from pipecat.processors.aggregators.llm_context import LLMContext


async def _resolve_model_configuration(organization_id, model_configuration_uuid=None):
    if model_configuration_uuid is None:
        return (
            await get_resolved_ai_model_configuration(organization_id=organization_id)
        ).effective
    resolved = await resolve_model_configuration(
        organization_id,
        api_override={"model_configuration_uuid": model_configuration_uuid},
    )
    return resolved.effective


async def generate_workflow(
    *,
    organization_id: int,
    call_type: Literal["inbound", "outbound"],
    use_case: str,
    activity_description: str,
    model_configuration_uuid: str | None = None,
) -> dict | None:
    """Generate only workflow data, returning None for the managed-service path."""
    configuration = await _resolve_model_configuration(
        organization_id, model_configuration_uuid
    )
    if (
        configuration.llm is None
        or configuration.llm.provider == ServiceProviders.DOGRAH.value
    ):
        return None

    llm = create_llm_service_with_model_override(
        configuration, None, usage_context="workflow_template"
    )
    context = LLMContext()
    context.set_messages(
        [
            {
                "role": "user",
                "content": (
                    f"Call type: {call_type}\nUse case: {use_case}\n"
                    f"Activity description: {activity_description}\n\n"
                    "Create a concise, useful voice-agent workflow with a greeting, "
                    "an agent node, and an end node. Return only a JSON object with "
                    "an optional name and a workflow_definition containing nodes and edges. "
                    "Node IDs and edge IDs must be unique. Agent prompts must be specific "
                    "to this use case. Use startCall, agentNode and endCall node types; use "
                    "the same node IDs in edge source and target. Do not invent knowledge-base "
                    "documents, tools, or integrations."
                ),
            }
        ]
    )
    response = await llm.run_inference(
        context,
        max_tokens=6000,
        system_instruction=(
            "You design voice-agent workflows. Return only the requested JSON, with "
            "node positions and with each node's name and prompt in its data object."
        ),
    )
    result = parse_llm_json(response or "", raise_on_error=True)
    if not isinstance(result, dict) or not isinstance(
        result.get("workflow_definition"), dict
    ):
        raise ValueError(
            "Configured workflow provider returned an invalid workflow definition"
        )
    if not isinstance(
        result["workflow_definition"].get("nodes"), list
    ) or not isinstance(result["workflow_definition"].get("edges"), list):
        raise ValueError(
            "Configured workflow provider returned an invalid workflow definition"
        )
    return result


async def transcribe_audio(
    *,
    organization_id: int,
    audio_data: bytes,
    language: str,
    model_configuration_uuid: str | None = None,
) -> dict | None:
    """Transcribe WAV bytes through the configured OpenAI STT service."""
    if not audio_data.startswith(b"RIFF") or audio_data[8:12] != b"WAVE":
        return None
    configuration = await _resolve_model_configuration(
        organization_id, model_configuration_uuid
    )
    if (
        configuration.stt is None
        or configuration.stt.provider == ServiceProviders.DOGRAH.value
    ):
        return None
    if configuration.stt.provider != ServiceProviders.OPENAI.value:
        # Existing STT providers have different streaming/audio contracts. Do not
        # claim that their in-pipeline services can process an uploaded audio file.
        return None

    audio_config = AudioConfig(
        transport_in_sample_rate=16000,
        transport_out_sample_rate=16000,
    )
    stt = create_stt_service(configuration, audio_config)
    stt._settings.language = None if language == "multi" else language
    result = await stt._transcribe(audio_data)
    transcript = (getattr(result, "text", None) or "").strip()
    if not transcript:
        raise HTTPException(
            status_code=422, detail="The configured STT provider returned no transcript"
        )
    return {"transcript": transcript}
