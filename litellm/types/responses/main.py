from typing import List, Literal, Optional, Union

from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
from pydantic import PrivateAttr
from typing_extensions import Any, List, Optional, TypedDict

from litellm.types.llms.base import BaseLiteLLMOpenAIResponseObject

Phase = Optional[Literal["commentary", "final_answer"]]


class GenericResponseOutputItemContentAnnotation(BaseLiteLLMOpenAIResponseObject):
    """Annotation for content in a message"""

    type: Optional[str]
    start_index: Optional[int]
    end_index: Optional[int]
    url: Optional[str]
    title: Optional[str]
    pass


class OutputText(BaseLiteLLMOpenAIResponseObject):
    """Text output content from an assistant message"""

    type: Optional[str]  # "output_text"
    text: Optional[str]
    annotations: Optional[List[GenericResponseOutputItemContentAnnotation]]


class OutputFunctionToolCall(BaseLiteLLMOpenAIResponseObject):
    """A tool call to run a function"""

    arguments: Optional[str]
    call_id: Optional[str]
    name: Optional[str]
    type: Optional[str]  # "function_call"
    id: Optional[str]
    status: Literal["in_progress", "completed", "incomplete"]
    phase: Phase = None


class OutputImageGenerationCall(BaseLiteLLMOpenAIResponseObject):
    """An image generation call output"""

    type: Literal["image_generation_call"]
    id: str
    status: Literal["in_progress", "completed", "incomplete", "failed"]
    result: Optional[str]  # Base64 encoded image data (without data:image prefix)


class OutputCodeInterpreterCallLog(BaseLiteLLMOpenAIResponseObject):
    """Log output from a code interpreter call"""

    type: Literal["logs"]
    logs: str


class OutputCodeInterpreterCall(BaseLiteLLMOpenAIResponseObject):
    """A code interpreter / code execution call output"""

    type: Literal["code_interpreter_call"]
    id: str
    code: Optional[str]
    container_id: Optional[str]
    status: Literal["in_progress", "completed", "incomplete", "failed"]
    outputs: Optional[List[OutputCodeInterpreterCallLog]]


class OutputWebSearchCallAction(BaseLiteLLMOpenAIResponseObject):
    """Search action for a Responses API web_search_call item"""

    type: Literal["search"]
    query: Optional[str] = None


class OutputWebSearchCall(BaseLiteLLMOpenAIResponseObject):
    """A server-side web search call surfaced in Responses API output.

    Anthropic/Vertex native web_search arrives as chat tool_calls with
    srvtoolu_* ids. Emitting those as generic function_call items breaks
    multi-turn replay (clients treat them as unanswered function calls).
    """

    type: Literal["web_search_call"]
    id: str
    status: Literal["in_progress", "completed", "incomplete", "failed"]
    action: Optional[OutputWebSearchCallAction] = None


def build_code_interpreter_log_outputs(
    content: Any,
) -> Optional[List[OutputCodeInterpreterCallLog]]:
    """Convert Anthropic bash_code_execution stdout/stderr to log outputs.

    Shared by streaming (handler.py) and non-streaming (transformation.py) paths.
    """
    if not isinstance(content, dict):
        return None
    parts = []
    if content.get("stdout"):
        parts.append(content["stdout"])
    if content.get("stderr"):
        parts.append(f"STDERR: {content['stderr']}")
    logs = "".join(parts)
    return [OutputCodeInterpreterCallLog(type="logs", logs=logs)] if logs else None


class CustomToolCallOutputItem(BaseLiteLLMOpenAIResponseObject):
    """A custom/freeform tool call output item (e.g. apply_patch).

    Mirrors the ``custom_tool_call`` variant of OpenAI's Responses API output.
    Unlike ``OutputFunctionToolCall`` which uses ``arguments`` (JSON string),
    this uses ``input`` (raw string) for the tool payload.
    """

    type: Literal["custom_tool_call"]
    call_id: str
    id: Optional[str] = None
    name: str
    input: str
    status: Optional[Literal["in_progress", "completed", "incomplete"]] = None


class GenericResponseOutputItem(BaseLiteLLMOpenAIResponseObject):
    """
    Generic response API output item

    """

    type: str  # "message"
    id: str
    status: str  # "completed", "in_progress", etc.
    role: str  # "assistant", "user", etc.
    content: List[OutputText]
    phase: Phase = None


class DeleteResponseResult(BaseLiteLLMOpenAIResponseObject):
    """
    Result of a delete response request

    {
        "id": "resp_6786a1bec27481909a17d673315b29f6",
        "object": "response",
        "deleted": true
    }
    """

    id: Optional[str]
    object: Optional[str]
    deleted: Optional[bool]

    # Define private attributes using PrivateAttr
    _hidden_params: dict = PrivateAttr(default_factory=dict)


class DecodedResponseId(TypedDict, total=False):
    """Structure representing a decoded response ID"""

    custom_llm_provider: Optional[str]
    model_id: Optional[str]
    response_id: str
