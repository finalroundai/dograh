"""Provider inference for knowledge-base text extraction and chunking.

Only UTF-8 text and Markdown documents have a configured-provider extraction
path. Existing document formats continue to use MPS; configured embeddings
remain a separate runtime service in the ingestion task.
"""

from api.services.configuration.registry import ServiceProviders
from api.services.gen_ai.json_parser import parse_llm_json
from api.services.pipecat.service_factory import create_llm_service_with_model_override
from api.services.provider_utility_inference import _resolve_model_configuration
from pipecat.processors.aggregators.llm_context import LLMContext


async def process_text_document(
    *,
    organization_id: int,
    file_path: str,
    filename: str,
    retrieval_mode: str,
    max_tokens: int,
    model_configuration_uuid: str | None = None,
) -> dict | None:
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in (".txt", ".md"):
        return None
    configuration = await _resolve_model_configuration(
        organization_id, model_configuration_uuid
    )
    if (
        configuration.llm is None
        or configuration.llm.provider == ServiceProviders.DOGRAH.value
    ):
        return None

    with open(file_path, encoding="utf-8-sig") as document:
        text = document.read()
    context = LLMContext()
    context.set_messages(
        [
            {
                "role": "user",
                "content": (
                    "Extract and index the document below. Return only a JSON object with "
                    "full_text and chunks. full_text must be the complete document. chunks "
                    f"must contain semantically coherent chunks of about {max_tokens} tokens, "
                    "each with chunk_text and a short contextualized_text explaining the chunk's "
                    "place in the document. Use an empty chunks array when full_document mode is "
                    f"specified. Retrieval mode: {retrieval_mode}\n\n{text}"
                ),
            }
        ]
    )
    llm = create_llm_service_with_model_override(
        configuration, None, usage_context="document_extraction"
    )
    response = await llm.run_inference(
        context,
        system_instruction=(
            "Extract and index only the supplied document. Preserve document text exactly; "
            "do not add facts or omit material from full_text. Return only JSON."
        ),
    )
    result = parse_llm_json(response or "", raise_on_error=True)
    if not isinstance(result, dict) or not isinstance(result.get("full_text"), str):
        raise ValueError("Configured document provider returned an invalid extraction")
    chunks = result.get("chunks")
    if retrieval_mode == "full_document":
        return {"full_text": result["full_text"], "docling_metadata": {}}
    if not isinstance(chunks, list):
        raise ValueError("Configured document provider returned an invalid chunk list")
    validated_chunks = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError("Configured document provider returned an invalid chunk")
        if not all(
            isinstance(chunk.get(key), str)
            for key in ("chunk_text", "contextualized_text")
        ):
            raise ValueError("Configured document provider returned an invalid chunk")
        metadata = chunk.get("chunk_metadata")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError(
                "Configured document provider returned invalid chunk metadata"
            )
        validated_chunks.append(
            {
                "chunk_text": chunk["chunk_text"],
                "contextualized_text": chunk["contextualized_text"],
                "chunk_index": len(validated_chunks),
                "chunk_metadata": metadata or {},
                "token_count": chunk.get("token_count", 0),
            }
        )
    if result["full_text"].strip() and not validated_chunks:
        raise ValueError("Configured document provider returned no usable chunks")
    return {"chunks": validated_chunks, "docling_metadata": {}}
