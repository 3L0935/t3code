# Ollama Tool Calling — Design Spec

**Date:** 2026-05-16  
**Scope:** Add native function-calling tool support to the Ollama provider, matching the approval flow of the other providers (Claude, Cursor, OpenCode).

---

## Context

The Ollama provider was added as a pure-chat adapter (no tool use). To make it useful as a coding assistant, the model needs to be able to read files, write files, run shell commands, and search the codebase — the same operations Claude performs via the claude-agent-sdk.

Ollama exposes an OpenAI-compatible `/api/chat` endpoint that accepts a `tools` array. Models that support function calling (llama3.1, llama3.2, qwen2.5, mistral-nemo, etc.) respond with `tool_calls` in the assistant message when they want to invoke a tool.

---

## Quick Fix: OllamaSettings `model` field

`KeybindingsToast.browser.tsx` (fixture) references a `model` field that was omitted from the `OllamaSettings` schema in `settings.ts`. Add it:

```ts
model: TrimmedString.pipe(
  Schema.withDecodingDefault(Effect.succeed("")),
  Schema.annotateKey({
    title: "Default model",
    description: "Default model slug. Overridden by per-session model selection.",
    providerSettingsForm: { placeholder: "qwen2.5:7b", clearWhenEmpty: "omit" },
  }),
),
```

Also add `model: Schema.optionalKey(TrimmedString)` to `OllamaSettingsPatch`. This fixes the typecheck failure and lets users set a default model.

---

## Tool Definitions

Five tools, defined as OpenAI-compatible JSON schemas in a new module `OllamaTools.ts`:

| Tool | Description | Key params |
|---|---|---|
| `read_file` | Read file content | `path`, optional `offset` (line), `limit` (lines) |
| `write_file` | Create or overwrite a file | `path`, `content` |
| `bash` | Run a shell command | `command`, optional `cwd` |
| `list_directory` | List files in a directory | `path`, optional `recursive` (bool) |
| `search_files` | Search for a pattern in files (grep) | `pattern`, optional `path`, `file_glob` |

Each tool maps to a `CanonicalItemType`:
- `bash` → `command_execution`
- `write_file` → `file_change`
- `read_file` → `dynamic_tool_call` (read-only, mapped to `file_read_approval`)
- `list_directory`, `search_files` → `dynamic_tool_call`

---

## Architecture

### New file: `apps/server/src/provider/OllamaTools.ts`

- Exports `OLLAMA_TOOL_DEFINITIONS: readonly OllamaToolDefinition[]` — the 5 JSON schemas
- Exports `executeOllamaTool(call: OllamaToolCall, cwd: string): Effect<string, OllamaToolError>` — executes the tool server-side using Effect's `FileSystem` and `Command` services
- Exports `classifyOllamaToolItemType(toolName: string): CanonicalItemType` — mirrors the Claude adapter's `classifyToolItemType` logic
- Exports `classifyOllamaRequestType(toolName: string): CanonicalRequestType`
- Exports `summarizeOllamaToolCall(toolName: string, args: Record<string, unknown>): string`

### Modified: `apps/server/src/provider/ollamaRuntime.ts`

Add types and update the chat functions to support tools:

```ts
interface OllamaToolDefinition { type: "function"; function: { name, description, parameters } }
interface OllamaToolCall { function: { name: string; arguments: Record<string, unknown> } }

// ollamaChatStream / ollamaChat gain an optional `tools?` param
// Chunks/responses gain an optional `tool_calls?: OllamaToolCall[]`
```

### Modified: `apps/server/src/provider/Layers/OllamaAdapter.ts`

**Session context** gains:
```ts
readonly pendingApprovals: Map<ApprovalRequestId, Deferred<ProviderApprovalDecision>>
```

**`sendTurn`** becomes a loop:
1. Push user message, POST `/api/chat` with `OLLAMA_TOOL_DEFINITIONS`
2. If response has `tool_calls`:
   a. Emit `item.started` (with `classifyOllamaToolItemType`)
   b. Check `runtimeMode`:
      - `full-access` → auto-approve
      - else → emit `request.opened` + `Deferred.await`
   c. Execute via `executeOllamaTool`
   d. Emit `item.completed` with result
   e. Append to history: `{role:"assistant", tool_calls:[...]}` then `{role:"tool", content: result}`
   f. Goto 1
3. Otherwise stream `content.delta` → `item.completed` → `turn.completed`

**`respondToRequest`** (currently no-op) resolves the pending `Deferred` for the given `requestId`.

**`OllamaAdapter` needs** `Effect.FileSystem` and `Effect.Command` in its service environment for tool execution.

### Modified: `packages/contracts/src/settings.ts`

Add `model` field to `OllamaSettings` and `OllamaSettingsPatch` (see Quick Fix above).

---

## Data Flow

```
User sends message
  → OllamaAdapter.sendTurn
    → POST /api/chat { messages, tools: OLLAMA_TOOL_DEFINITIONS }
    → response.tool_calls?
        YES:
          emit item.started (dynamic_tool_call | command_execution | file_change)
          runtimeMode == full-access?
            YES: execute immediately
            NO:  emit request.opened → Deferred.await(decision)
                 [user approves/rejects via respondToRequest]
          execute tool → emit item.completed
          append tool result to messages
          loop ↑
        NO:
          stream content.delta chunks
          emit item.completed (assistant_message)
          emit turn.completed
```

---

## Error Handling

- Tool execution errors (file not found, command failed) → result string includes the error, fed back to the model (the model can try to recover)
- Ollama API errors → `turn.completed { state: "failed" }` + `runtime.error`
- User rejects a tool call → stop the loop, emit `turn.completed { state: "interrupted" }`

---

## Files Touched

| File | Change |
|---|---|
| `packages/contracts/src/settings.ts` | Add `model` to OllamaSettings + OllamaSettingsPatch |
| `apps/server/src/provider/ollamaRuntime.ts` | Add tool types, extend chat functions |
| `apps/server/src/provider/OllamaTools.ts` | New — tool schemas + execution |
| `apps/server/src/provider/Layers/OllamaAdapter.ts` | Tool loop in sendTurn, fix respondToRequest |

No web/UI changes needed — the existing approval UI already handles `request.opened` events generically.
