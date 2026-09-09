/**
 * Ollama runtime utilities — HTTP helpers for the Ollama REST API.
 *
 * All calls go through the Effect `HttpClient`. Chat streaming pipes the
 * response body through a line reader for NDJSON chunks.
 *
 * @module provider/ollamaRuntime
 */
import * as Data from "effect/Data";
import * as Effect from "effect/Effect";
import * as Schema from "effect/Schema";
import * as Stream from "effect/Stream";
import { HttpClient, HttpClientRequest, HttpClientResponse } from "effect/unstable/http";

// ── Types ──────────────────────────────────────────────────────────────

export type OllamaChatRole = "system" | "user" | "assistant" | "tool";

export interface OllamaToolDefinition {
  readonly type: "function";
  readonly function: {
    readonly name: string;
    readonly description: string;
    readonly parameters: Record<string, unknown>;
  };
}

export interface OllamaToolCall {
  readonly function: {
    readonly name: string;
    readonly arguments: Record<string, unknown>;
  };
}

export interface OllamaChatMessage {
  readonly role: OllamaChatRole;
  readonly content: string;
  readonly tool_calls?: readonly OllamaToolCall[];
  /** Base64-encoded image data (Ollama multimodal input). */
  readonly images?: readonly string[];
}

export interface OllamaChatResponse {
  readonly model: string;
  readonly createdAt: string;
  readonly message: OllamaChatMessage;
  readonly done: boolean;
  readonly doneReason?: string;
  readonly totalDuration?: number;
  readonly promptEvalCount?: number;
  readonly evalCount?: number;
}

export interface OllamaChatChunk {
  readonly model: string;
  readonly createdAt: string;
  readonly message: { readonly role: OllamaChatRole; readonly content: string; readonly tool_calls?: readonly OllamaToolCall[] };
  readonly done: boolean;
  readonly doneReason?: string;
  readonly promptEvalCount?: number;
  readonly evalCount?: number;
}

export interface OllamaModelInfo {
  readonly name: string;
  readonly modifiedAt: string;
  readonly size: number;
  readonly digest: string;
}

// ── Error ──────────────────────────────────────────────────────────────

const RUNTIME_ERROR_TAG = "OllamaRuntimeError";

export class OllamaRuntimeError extends Data.TaggedError(RUNTIME_ERROR_TAG)<{
  readonly operation: string;
  readonly detail: string;
  readonly cause?: unknown;
}> {
  static readonly is = (u: unknown): u is OllamaRuntimeError =>
    typeof u === "object" && u !== null && (u as Record<string, unknown>)._tag === RUNTIME_ERROR_TAG;
}

function fail(operation: string, detail: string, cause?: unknown): Effect.Effect<never, OllamaRuntimeError> {
  return Effect.fail(new OllamaRuntimeError({ operation, detail, cause }));
}

// ── Headers helper ─────────────────────────────────────────────────────

function buildHeaders(apiKey?: string): Record<string, string> {
  const headers: Record<string, string> = { "Content-Type": "application/json", Accept: "application/x-ndjson" };
  if (apiKey && apiKey.trim().length > 0) {
    headers["Authorization"] = `Bearer ${apiKey}`;
  }
  return headers;
}

// ── Chunk decoding ─────────────────────────────────────────────────────

const ChatChunkSchema = Schema.Struct({
  model: Schema.optional(Schema.String),
  created_at: Schema.optional(Schema.String),
  message: Schema.Struct({
    role: Schema.optional(Schema.String),
    content: Schema.optional(Schema.String),
    tool_calls: Schema.optional(Schema.Unknown),
  }),
  done: Schema.optional(Schema.Boolean),
  done_reason: Schema.optional(Schema.String),
  prompt_eval_count: Schema.optional(Schema.Number),
  eval_count: Schema.optional(Schema.Number),
});
// Chunk lines arrive as JSON strings; fromJsonString parses before decoding.
const JsonChunkLine = Schema.fromJsonString(ChatChunkSchema);
const decodeChunkLine = Schema.decodeUnknownSync(JsonChunkLine);

/** Test-only escape hatch: parse one NDJSON chat line. */
export const parseChunkLineForTest = parseChatChunk;

function parseChatChunk(line: string): OllamaChatChunk {
  const raw = decodeChunkLine(line);
  return {
    model: raw.model ?? "",
    createdAt: raw.created_at ?? "",
    message: {
      role: (raw.message.role ?? "assistant") as OllamaChatRole,
      content: raw.message.content ?? "",
      ...(Array.isArray(raw.message.tool_calls) ? { tool_calls: raw.message.tool_calls as readonly OllamaToolCall[] } : {}),
    },
    done: raw.done === true,
    ...(raw.done_reason ? { doneReason: raw.done_reason } : {}),
    ...(typeof raw.prompt_eval_count === "number" ? { promptEvalCount: raw.prompt_eval_count } : {}),
    ...(typeof raw.eval_count === "number" ? { evalCount: raw.eval_count } : {}),
  };
}

// ── Non-streaming chat ─────────────────────────────────────────────────

const ChatResponseSchema = Schema.Struct({
  model: Schema.String,
  created_at: Schema.optional(Schema.String),
  message: Schema.Struct({
    role: Schema.String,
    content: Schema.optional(Schema.String),
    tool_calls: Schema.optional(Schema.Unknown),
  }),
  done: Schema.Boolean,
  done_reason: Schema.optional(Schema.String),
  total_duration: Schema.optional(Schema.Number),
  prompt_eval_count: Schema.optional(Schema.Number),
  eval_count: Schema.optional(Schema.Number),
});
const decodeChatResponse = Schema.decodeUnknownSync(ChatResponseSchema);

export const ollamaChat = (input: {
  readonly client: HttpClient.HttpClient;
  readonly baseUrl: string;
  readonly apiKey?: string | undefined;
  readonly model: string;
  readonly messages: ReadonlyArray<OllamaChatMessage>;
  readonly tools?: ReadonlyArray<OllamaToolDefinition>;
  readonly options?: Record<string, unknown>;
}) =>
  Effect.gen(function* () {
    const client = input.client;
    const request = HttpClientRequest.post(`${input.baseUrl}/api/chat`).pipe(
      HttpClientRequest.setHeaders(buildHeaders(input.apiKey)),
      HttpClientRequest.bodyJsonUnsafe({
        model: input.model,
        messages: [...input.messages],
        stream: false,
        ...(input.tools && input.tools.length > 0 ? { tools: [...input.tools] } : {}),
        ...(input.options ? { options: input.options } : {}),
      }),
    );
    const response = yield* client.execute(request).pipe(
      Effect.mapError(
        (cause) => new OllamaRuntimeError({ operation: "ollamaChat", detail: cause.message, cause }),
      ),
    );
    if (response.status < 200 || response.status >= 300) {
      const text = yield* response.text.pipe(
        Effect.mapError((cause) => new OllamaRuntimeError({ operation: "ollamaChat.text", detail: cause.message, cause })),
      );
      return yield* fail("ollamaChat", `Ollama /api/chat returned status ${response.status}: ${text}`);
    }
    const json = yield* response.json.pipe(
      Effect.mapError((cause) => new OllamaRuntimeError({ operation: "ollamaChat.json", detail: cause.message, cause })),
    );
    let parsed: ReturnType<typeof decodeChatResponse>;
    try {
      parsed = decodeChatResponse(json);
    } catch (cause) {
      return yield* fail("ollamaChat", "Ollama /api/chat returned unexpected response shape.", cause);
    }
    return {
      model: parsed.model,
      createdAt: parsed.created_at ?? "",
      message: {
        role: parsed.message.role as OllamaChatRole,
        content: parsed.message.content ?? "",
        ...(Array.isArray(parsed.message.tool_calls) ? { tool_calls: parsed.message.tool_calls as readonly OllamaToolCall[] } : {}),
      },
      done: parsed.done,
      ...(parsed.done_reason ? { doneReason: parsed.done_reason } : {}),
      ...(typeof parsed.total_duration === "number" ? { totalDuration: parsed.total_duration } : {}),
      ...(typeof parsed.prompt_eval_count === "number" ? { promptEvalCount: parsed.prompt_eval_count } : {}),
      ...(typeof parsed.eval_count === "number" ? { evalCount: parsed.eval_count } : {}),
    } satisfies OllamaChatResponse;
  });

// ── Streaming chat (NDJSON via HttpClient) ─────────────────────────────

export const ollamaChatStream = (input: {
  readonly client: HttpClient.HttpClient;
  readonly baseUrl: string;
  readonly apiKey?: string | undefined;
  readonly model: string;
  readonly messages: ReadonlyArray<OllamaChatMessage>;
  readonly tools?: ReadonlyArray<OllamaToolDefinition>;
  readonly options?: Record<string, unknown>;
}): Stream.Stream<OllamaChatChunk, OllamaRuntimeError> =>
  Stream.unwrap(
    Effect.gen(function* () {
      const client = input.client;
      const request = HttpClientRequest.post(`${input.baseUrl}/api/chat`).pipe(
        HttpClientRequest.setHeaders({ ...buildHeaders(input.apiKey), Accept: "application/x-ndjson" }),
        HttpClientRequest.bodyJsonUnsafe({
          model: input.model,
          messages: [...input.messages],
          stream: true,
          ...(input.tools && input.tools.length > 0 ? { tools: [...input.tools] } : {}),
          ...(input.options ? { options: input.options } : {}),
        }),
      );
      const response = yield* client.execute(request).pipe(
        Effect.mapError(
          (cause) => new OllamaRuntimeError({ operation: "ollamaChatStream", detail: cause.message, cause }),
        ),
      );
      if (response.status < 200 || response.status >= 300) {
        const text = yield* response.text.pipe(
          Effect.mapError((cause) => new OllamaRuntimeError({ operation: "ollamaChatStream.text", detail: cause.message, cause })),
        );
        return yield* fail("ollamaChatStream", `Ollama /api/chat stream returned status ${response.status}: ${text}`);
      }
      return response.stream.pipe(
        Stream.decodeText(),
        Stream.splitLines,
        Stream.filter((line) => line.trim().length > 0),
        Stream.mapError(
          (cause) => new OllamaRuntimeError({ operation: "ollamaChatStream.read", detail: cause.message, cause }),
        ),
        Stream.mapEffect((line) =>
          Effect.try({
            try: () => parseChatChunk(line),
            catch: (cause: unknown) =>
              new OllamaRuntimeError({
                operation: "ollamaChatStream.parse",
                detail: `Failed to parse SSE line: ${String(line).slice(0, 200)}`,
                cause,
              }),
          }),
        ),
      );
    }),
  );

// ── Model listing ──────────────────────────────────────────────────────

const ListModelsSchema = Schema.Struct({
  models: Schema.Array(
    Schema.Struct({
      name: Schema.String,
      modified_at: Schema.optional(Schema.String),
      size: Schema.optional(Schema.Number),
      digest: Schema.optional(Schema.String),
    }),
  ),
});
const decodeListModels = Schema.decodeUnknownSync(ListModelsSchema);

export const ollamaListModels = (client: HttpClient.HttpClient, baseUrl: string, apiKey?: string) =>
  Effect.gen(function* () {
    const request = HttpClientRequest.get(`${baseUrl}/api/tags`).pipe(HttpClientRequest.setHeaders(buildHeaders(apiKey)));
    const response = yield* client.execute(request).pipe(
      Effect.mapError(
        (cause) => new OllamaRuntimeError({ operation: "ollamaListModels", detail: cause.message, cause }),
      ),
    );
    if (response.status < 200 || response.status >= 300) {
      const text = yield* response.text.pipe(
        Effect.mapError((cause) => new OllamaRuntimeError({ operation: "ollamaListModels.text", detail: cause.message, cause })),
      );
      return yield* fail("ollamaListModels", `Ollama /api/tags returned status ${response.status}: ${text}`);
    }
    const json = yield* response.json.pipe(
      Effect.mapError((cause) => new OllamaRuntimeError({ operation: "ollamaListModels.json", detail: cause.message, cause })),
    );
    let parsed: typeof ListModelsSchema.Type;
    try {
      parsed = decodeListModels(json);
    } catch (cause) {
      return yield* fail("ollamaListModels", "Ollama /api/tags returned unexpected response shape.", cause);
    }
    return parsed.models.map((m) => ({
      name: m.name,
      modifiedAt: m.modified_at ?? "",
      size: typeof m.size === "number" ? m.size : 0,
      digest: m.digest ?? "",
    })) satisfies ReadonlyArray<OllamaModelInfo>;
  });

// ── Version check ──────────────────────────────────────────────────────

const VersionSchema = Schema.Struct({
  version: Schema.optional(Schema.String),
});
const decodeVersion = Schema.decodeUnknownSync(VersionSchema);

export const ollamaVersion = (client: HttpClient.HttpClient, baseUrl: string, apiKey?: string) =>
  Effect.gen(function* () {
    const request = HttpClientRequest.get(`${baseUrl}/api/version`).pipe(HttpClientRequest.setHeaders(buildHeaders(apiKey)));
    const response = yield* client.execute(request).pipe(
      Effect.mapError(
        () => new OllamaRuntimeError({ operation: "ollamaVersion", detail: "Failed to reach Ollama /api/version" }),
      ),
    );
    if (response.status < 200 || response.status >= 300) return "";
    const json = yield* response.json.pipe(
      Effect.mapError(
        () => new OllamaRuntimeError({ operation: "ollamaVersion.json", detail: "Failed to parse /api/version response" }),
      ),
    );
    try {
      return decodeVersion(json).version ?? "";
    } catch {
      return "";
    }
  });