"""OpenAI 兼容的请求/响应 Pydantic 模型 (demo 版, 只保留必要字段)."""
from __future__ import annotations
from typing import Literal, Optional, Union
from pydantic import BaseModel, Field


# ============ Chat Completion ============

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 1.0
    top_p: float = 1.0                      # 接住但不用
    max_tokens: Optional[int] = None
    stream: bool = False
    stop: Union[str, list[str], None] = None  # 接住但不用
    presence_penalty: float = 0.0             # 接住但不用
    frequency_penalty: float = 0.0            # 接住但不用
    n: int = 1                                # 只支持 1
    user: Optional[str] = None


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo


# ============ Streaming Chunk ============

class DeltaMessage(BaseModel):
    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None


class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionStreamChoice]


# ============ /v1/models ============

class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "nanovllm"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


# ============ Error ============

class ErrorResponse(BaseModel):
    error: dict = Field(..., description="{message, type, code}")
