import * as DateTime from "effect/DateTime";
import * as Crypto from "effect/Crypto";
import * as Deferred from "effect/Deferred";
import * as Effect from "effect/Effect";
import * as Exit from "effect/Exit";
import * as Fiber from "effect/Fiber";
import * as FileSystem from "effect/FileSystem";
import * as Option from "effect/Option";
import * as Queue from "effect/Queue";
import * as Stream from "effect/Stream";
import { HttpClient } from "effect/unstable/http";
import {
  ApprovalRequestId,
  EventId,
  OllamaSettings,
  ProviderDriverKind,
  ProviderInstanceId,
  RuntimeRequestId,
  TurnTokenUsage,
  type ProviderApprovalDecision,
  type ProviderRuntimeEvent,
  type ProviderSession,
  type ProviderSessionStartInput,
  type ProviderSendTurnInput,
  type ProviderTurnStartResult,
  RuntimeItemId,
  ThreadId,
  TurnId,
} from "@t3tools/contracts";

import type { OllamaAdapterShape } from "../Services/OllamaAdapter.ts";
import {
  ProviderAdapterRequestError,
  ProviderAdapterSessionNotFoundError,
  ProviderAdapterValidationError,
} from "../Errors.ts";
import { ServerConfig } from "../../config.ts";
import { resolveAttachmentPath } from "../../attachmentStore.ts";
import { buildRuntimeInstructions } from "../RuntimeInstructions.ts";
import { ollamaChatStream, type OllamaChatMessage, type OllamaRuntimeError } from "../ollamaRuntime.js";
import {
  OLLAMA_TOOL_DEFINITIONS,
  executeOllamaTool,
  classifyOllamaToolItemType,
  classifyOllamaRequestType,
  summarizeOllamaToolCall,
} from "../OllamaTools.js";
import { type EventNdjsonLogger } from "./EventNdjsonLogger.ts";

const PROVIDER = ProviderDriverKind.make("ollama");

/** Default model used when neither the session nor settings name one. Kept in
 * sync with `DEFAULT_MODEL_BY_PROVIDER["ollama"]` in contracts. */
const FALLBACK_MODEL = "qwen2.5:7b";

interface PendingApproval {
  readonly requestType: string;
  readonly detail: string;
  readonly decision: Deferred.Deferred<ProviderApprovalDecision>;
}

/** Per-turn token usage accumulated from Ollama chunk counters. */
interface TurnUsageState {
  promptTokens: number;
  responseTokens: number;
  present: boolean;
}

interface OllamaSessionContext {
  session: ProviderSession;
  readonly threadId: ThreadId;
  readonly messages: OllamaChatMessage[];
  readonly runtimeEvents: Queue.Queue<ProviderRuntimeEvent>;
  readonly stopped: { current: boolean };
  readonly pendingApprovals: Map<ApprovalRequestId, PendingApproval>;
  activeModel: string;
  activeTurnId: TurnId | undefined;
  activeFiber: Fiber.Fiber<unknown, unknown> | undefined;
  /** message count at the start of each turn (indexed by turn number, 0 = before first user message) */
  readonly turnMessageIndices: number[];
}

const nowIso = Effect.map(DateTime.now, DateTime.formatIso);

export const makeOllamaAdapter = (
  ollamaSettings: OllamaSettings,
  processEnv: Record<string, string | undefined>,
  options?: {
    readonly instanceId?: ProviderInstanceId;
    readonly nativeEventLogger?: EventNdjsonLogger;
  },
) =>
  Effect.gen(function* () {
    const boundInstanceId = options?.instanceId ?? ProviderInstanceId.make("ollama");
    const runtimeEvents = yield* Queue.unbounded<ProviderRuntimeEvent>();
    const sessions = new Map<ThreadId, OllamaSessionContext>();
    const apiKey = processEnv.OLLAMA_API_KEY;
    const crypto = yield* Effect.service(Crypto.Crypto);
    const httpClient = yield* HttpClient.HttpClient;
    const nativeEventLogger = options?.nativeEventLogger;

    yield* Effect.addFinalizer(() =>
      Effect.gen(function* () {
        for (const [, context] of sessions) {
          if (context.activeFiber) {
            context.stopped.current = true;
            yield* Fiber.interrupt(context.activeFiber).pipe(Effect.ignore);
          }
        }
        sessions.clear();
        yield* Queue.shutdown(runtimeEvents);
      }),
    );

    const randomUUIDv4 = crypto.randomUUIDv4.pipe(
      Effect.mapError(
        (cause) =>
          new ProviderAdapterRequestError({
            provider: PROVIDER,
            method: "crypto/randomUUIDv4",
            detail: "Failed to generate Ollama runtime identifier.",
            cause,
          }),
      ),
    );
    const makeEventStamp = () =>
      Effect.all({ eventId: Effect.map(randomUUIDv4, EventId.make), createdAt: nowIso });

    const emit = (event: ProviderRuntimeEvent) =>
      Queue.offer(runtimeEvents, event).pipe(Effect.asVoid);

    const logNative = (threadId: ThreadId, method: string, payload: unknown) =>
      Effect.gen(function* () {
        if (!nativeEventLogger) return;
        const observedAt = yield* nowIso;
        yield* nativeEventLogger.write(
          {
            observedAt,
            event: {
              id: yield* randomUUIDv4,
              kind: "notification",
              provider: PROVIDER,
              createdAt: observedAt,
              method,
              threadId,
              payload,
            },
          },
          threadId,
        );
      }).pipe(
        Effect.catchCause((cause) =>
          Effect.logWarning("Failed to write native Ollama event log.", { cause, threadId, method }),
        ),
      );

    const startSession: OllamaAdapterShape["startSession"] = Effect.fn("startSession")(
      function* (input: ProviderSessionStartInput) {
        sessions.delete(input.threadId);
        const createdAt = yield* nowIso;
        const effectiveModel =
          (input.modelSelection?.model?.trim().length ?? 0) > 0
            ? input.modelSelection!.model
            : ollamaSettings.model?.trim() || FALLBACK_MODEL;
        const session: ProviderSession = {
          provider: PROVIDER,
          providerInstanceId: boundInstanceId,
          status: "ready",
          runtimeMode: input.runtimeMode,
          cwd: input.cwd ?? process.cwd(),
          model: effectiveModel,
          threadId: input.threadId,
          createdAt,
          updatedAt: createdAt,
        };
        sessions.set(input.threadId, {
          session,
          threadId: input.threadId,
          messages: [],
          runtimeEvents,
          stopped: { current: false },
          pendingApprovals: new Map(),
          activeModel: effectiveModel,
          activeTurnId: undefined,
          activeFiber: undefined,
          turnMessageIndices: [0],
        });
        return session;
      },
    );

    const sendTurn: OllamaAdapterShape["sendTurn"] = Effect.fn("sendTurn")(
      function* (input: ProviderSendTurnInput) {
        const context = sessions.get(input.threadId);
        if (!context) {
          return yield* new ProviderAdapterSessionNotFoundError({ provider: PROVIDER, threadId: input.threadId });
        }
        const text = input.input?.trim();
        if (!text || text.length === 0) {
          return yield* new ProviderAdapterValidationError({ provider: PROVIDER, operation: "sendTurn", issue: "Ollama turns require text input." });
        }
        // A previous turn's fiber must be fully terminated before a new turn
        // starts, otherwise two fibers would mutate context.messages
        // concurrently. Fiber.interrupt awaits the fiber's onExit, so the
        // prior turn is fully closed (turn.completed emitted) once this returns.
        if (context.activeFiber) {
          yield* Fiber.interrupt(context.activeFiber);
          context.activeFiber = undefined;
        }
        // Clear any interrupt flag from a previous interruptTurn; stopSession
        // deletes the session entirely, so a closed session is already caught
        // by the not-found check above.
        context.stopped.current = false;
        const model = input.modelSelection?.model ?? context.activeModel;
        context.activeModel = model;
        const turnId = TurnId.make(`ollama-turn-${yield* randomUUIDv4}`);
        context.activeTurnId = turnId;
        // Record this turn's start boundary in the message array so
        // rollbackThread can splice whole turns (incl. tool messages),
        // regardless of whether the turn later succeeds or fails.
        const lastIndex = context.turnMessageIndices[context.turnMessageIndices.length - 1];
        if (lastIndex !== context.messages.length) {
          context.turnMessageIndices.push(context.messages.length);
        }

        // Image attachments ride on the user message as base64 `images`, the
        // only attachment channel Ollama supports. File attachments reach the
        // agent through the path line ProviderService puts in the prompt
        // (same as Grok).
        const serverConfigOption = yield* Effect.serviceOption(ServerConfig);
        const attachmentsDir = Option.isSome(serverConfigOption)
          ? serverConfigOption.value.attachmentsDir
          : undefined;
        const fileSystemOption = yield* Effect.serviceOption(FileSystem.FileSystem);
        const fileSystem = Option.isSome(fileSystemOption) ? fileSystemOption.value : undefined;
        const userMessage: { role: "user"; content: string; images?: ReadonlyArray<string> } = { role: "user", content: text };
        const images: Array<string> = [];
        for (const attachment of input.attachments ?? []) {
          if (attachment.type !== "image") continue;
          if (!attachmentsDir || !fileSystem) {
            return yield* new ProviderAdapterRequestError({
              provider: PROVIDER,
              method: "sendTurn",
              detail: "Image attachments are unavailable in this runtime.",
            });
          }
          const attachmentPath = resolveAttachmentPath({ attachmentsDir, attachment });
          if (!attachmentPath) {
            return yield* new ProviderAdapterRequestError({
              provider: PROVIDER,
              method: "sendTurn",
              detail: `Invalid attachment id '${attachment.id}'.`,
            });
          }
          const bytes = yield* fileSystem.readFile(attachmentPath).pipe(
            Effect.mapError(
              (cause) =>
                new ProviderAdapterRequestError({
                  provider: PROVIDER,
                  method: "sendTurn",
                  detail: "Failed to read attachment file.",
                  cause,
                }),
            ),
          );
          images.push(Buffer.from(bytes).toString("base64"));
        }
        if (images.length > 0) {
          userMessage.images = images;
        }
        context.messages.push(userMessage);
        context.session = { ...context.session, status: "running", activeTurnId: turnId, updatedAt: yield* nowIso };
        const cwd = context.session.cwd ?? process.cwd();
        const runtimeMode = context.session.runtimeMode ?? "full-access";
        const runtimeCtx = yield* Effect.context<never>();
        const runFork = Effect.runForkWith(runtimeCtx);

        yield* emit({
          ...(yield* makeEventStamp()),
          provider: PROVIDER,
          providerInstanceId: boundInstanceId,
          threadId: input.threadId,
          turnId,
          payload: { model },
          type: "turn.started",
        });

        yield* logNative(input.threadId, "ollama/turn/started", { turnId, model });

        // Captured by the ollamaChatStream catch below and read in onExit to
        // tell a real provider error apart from a fiber interruption. This
        // relies on the stream call being the loop's ONLY fallible step —
        // keep it that way, or onExit will misclassify an interruption as a
        // failure.
        let turnError: OllamaRuntimeError | undefined;
        // Accumulated across the turn's streaming rounds; read by onExit.
        const usage: TurnUsageState = { promptTokens: 0, responseTokens: 0, present: false };

        const runTurnLoop = Effect.gen(function* () {
          // Runtime instructions go first, once, before the first user
          // message — the same context every other harness injects. Re-run
          // after a rollback stripped the history back to an earlier turn.
          if (context.messages[0]?.role !== "system") {
            context.messages.unshift({ role: "system", content: buildRuntimeInstructions({ harness: "Ollama", model }) });
          }
          let looping = true;
          while (looping) {
            if (context.stopped.current) break;

            // One streaming completion. Content deltas are emitted as they
            // arrive; the last chunk carries done + token counters.
            let assistantItemId: RuntimeItemId | undefined;
            let assistantDeltaCount = 0;
            let assistantText = "";
            let finalMessage: OllamaChatMessage | undefined;
            const chunkStream = ollamaChatStream({
              client: httpClient,
              baseUrl: ollamaSettings.baseUrl,
              apiKey,
              model,
              messages: context.messages,
              tools: OLLAMA_TOOL_DEFINITIONS,
            }).pipe(
              Stream.catch((error: OllamaRuntimeError) => {
                turnError = error;
                return Stream.fail(error);
              }),
            );
            yield* Stream.runForEach(chunkStream, (chunk) =>
              Effect.gen(function* () {
                  if (typeof chunk.promptEvalCount === "number") usage.promptTokens += chunk.promptEvalCount;
                  if (typeof chunk.evalCount === "number") usage.responseTokens += chunk.evalCount;
                  if (chunk.promptEvalCount !== undefined || chunk.evalCount !== undefined) {
                    usage.present = true;
                  }
                  const delta = chunk.message.content ?? "";
                  if (delta.length > 0 && (!chunk.message.tool_calls || chunk.message.tool_calls.length === 0)) {
                    if (assistantItemId === undefined) {
                      assistantItemId = RuntimeItemId.make(`ollama:item:${turnId}:assistant`);
                      assistantDeltaCount = 0;
                      assistantText = "";
                      yield* emit({
                        ...(yield* makeEventStamp()),
                        provider: PROVIDER,
                        providerInstanceId: boundInstanceId,
                        threadId: input.threadId,
                        turnId,
                        itemId: assistantItemId,
                        type: "item.started",
                        payload: { itemType: "assistant_message", title: "Assistant message" },
                      });
                    }
                    assistantDeltaCount += 1;
                    assistantText += delta;
                    yield* emit({
                      ...(yield* makeEventStamp()),
                      provider: PROVIDER,
                      providerInstanceId: boundInstanceId,
                      threadId: input.threadId,
                      turnId,
                      itemId: assistantItemId,
                      type: "content.delta",
                      payload: { streamKind: "assistant_text", delta },
                    });
                  }
                  if (chunk.message.tool_calls && chunk.message.tool_calls.length > 0) {
                    finalMessage = chunk.message;
                  }
                  if (chunk.done) {
                    if (!finalMessage) finalMessage = chunk.message;
                  }
              }),
            );
            if (!finalMessage) {
              // Nothing usable came back this round.
              break;
            }

            if (finalMessage.tool_calls && finalMessage.tool_calls.length > 0) {
              // Close the open assistant text block before tool calls.
              if (assistantItemId !== undefined && assistantDeltaCount > 0) {
                yield* emit({
                  ...(yield* makeEventStamp()),
                  provider: PROVIDER,
                  providerInstanceId: boundInstanceId,
                  threadId: input.threadId,
                  turnId,
                  itemId: assistantItemId,
                  type: "item.completed",
                  payload: { itemType: "assistant_message", status: "completed", title: "Assistant message", detail: assistantText },
                });
                assistantItemId = undefined;
                assistantDeltaCount = 0;
                assistantText = "";
              }
              // Append assistant message with tool_calls to history
              context.messages.push({ role: "assistant", content: finalMessage.content ?? "", tool_calls: finalMessage.tool_calls });

              for (const toolCall of finalMessage.tool_calls) {
                if (context.stopped.current) {
                  looping = false;
                  break;
                }

                const toolName = toolCall.function.name;
                const toolArgs = toolCall.function.arguments;
                const itemType = classifyOllamaToolItemType(toolName);
                const requestType = classifyOllamaRequestType(toolName);
                const detail = summarizeOllamaToolCall(toolName, toolArgs);
                const itemId = `ollama:tool:${turnId}:${toolName}:${yield* randomUUIDv4}`;

                yield* emit({
                  ...(yield* makeEventStamp()),
                  provider: PROVIDER,
                  providerInstanceId: boundInstanceId,
                  threadId: input.threadId,
                  turnId,
                  itemId: RuntimeItemId.make(itemId),
                  type: "item.started",
                  payload: { itemType, title: detail },
                });

                let approved = true;
                if (runtimeMode !== "full-access") {
                  const requestId = ApprovalRequestId.make(yield* randomUUIDv4);
                  const decisionDeferred = yield* Deferred.make<ProviderApprovalDecision>();
                  context.pendingApprovals.set(requestId, { requestType, detail, decision: decisionDeferred });

                  yield* emit({
                    ...(yield* makeEventStamp()),
                    provider: PROVIDER,
                    providerInstanceId: boundInstanceId,
                    threadId: input.threadId,
                    turnId,
                    itemId: RuntimeItemId.make(itemId),
                    requestId: RuntimeRequestId.make(requestId),
                    type: "request.opened",
                    payload: { requestType, detail, args: { toolName, input: toolArgs } },
                  });

                  yield* logNative(input.threadId, "ollama/approval/requested", { turnId, toolName, requestId });

                  const decision = yield* Deferred.await(decisionDeferred);
                  context.pendingApprovals.delete(requestId);

                  yield* emit({
                    ...(yield* makeEventStamp()),
                    provider: PROVIDER,
                    providerInstanceId: boundInstanceId,
                    threadId: input.threadId,
                    turnId,
                    itemId: RuntimeItemId.make(itemId),
                    requestId: RuntimeRequestId.make(requestId),
                    type: "request.resolved",
                    payload: { requestType, decision },
                  });

                  if (decision === "cancel" || decision === "decline") {
                    approved = false;
                    looping = false;
                    yield* emit({
                      ...(yield* makeEventStamp()),
                      provider: PROVIDER,
                      providerInstanceId: boundInstanceId,
                      threadId: input.threadId,
                      turnId,
                      itemId: RuntimeItemId.make(itemId),
                      type: "item.completed",
                      payload: { itemType, status: "declined", title: detail },
                    });
                    break;
                  }
                }

                if (approved) {
                  const toolResult = yield* executeOllamaTool(toolCall, cwd).pipe(
                    Effect.provideService(HttpClient.HttpClient, httpClient),
                    Effect.catch((err) => Effect.succeed(`Error: ${err.detail}`)),
                  );

                  context.messages.push({ role: "tool", content: toolResult });

                  yield* emit({
                    ...(yield* makeEventStamp()),
                    provider: PROVIDER,
                    providerInstanceId: boundInstanceId,
                    threadId: input.threadId,
                    turnId,
                    itemId: RuntimeItemId.make(itemId),
                    type: "item.completed",
                    payload: { itemType, status: "completed", title: detail, detail: toolResult.slice(0, 500) || undefined },
                  });
                  yield* logNative(input.threadId, "ollama/tool/completed", { turnId, toolName, resultLength: toolResult.length });
                }
              }
            } else {
              // No tool calls → final assistant message; streaming deltas
              // were already emitted above.
              const content = finalMessage.content ?? "";
              context.messages.push({ role: "assistant", content });

              if (assistantItemId !== undefined && assistantDeltaCount > 0) {
                yield* emit({
                  ...(yield* makeEventStamp()),
                  provider: PROVIDER,
                  providerInstanceId: boundInstanceId,
                  threadId: input.threadId,
                  turnId,
                  itemId: assistantItemId,
                  type: "item.completed",
                  payload: { itemType: "assistant_message", status: "completed", title: "Assistant message", detail: assistantText },
                });
              } else if (content.length > 0) {
                const itemId = `ollama:item:${turnId}:assistant`;
                yield* emit({
                  ...(yield* makeEventStamp()),
                  provider: PROVIDER,
                  providerInstanceId: boundInstanceId,
                  threadId: input.threadId,
                  turnId,
                  itemId: RuntimeItemId.make(itemId),
                  type: "content.delta",
                  payload: { streamKind: "assistant_text", delta: content },
                });
                yield* emit({
                  ...(yield* makeEventStamp()),
                  provider: PROVIDER,
                  providerInstanceId: boundInstanceId,
                  threadId: input.threadId,
                  turnId,
                  itemId: RuntimeItemId.make(itemId),
                  type: "item.completed",
                  payload: { itemType: "assistant_message", status: "completed", title: "Assistant message", detail: content },
                });
              }

              looping = false;
            }
          }
        }).pipe(
          // onExit runs whether the loop completes, fails, or is interrupted,
          // so turn.completed is emitted exactly once on every path.
          Effect.onExit((exit) =>
            Effect.gen(function* () {
              context.activeTurnId = undefined;
              context.activeFiber = undefined;
              const tokenUsage: TurnTokenUsage = usage.present
                ? {
                    usageStatus: "complete",
                    usageScope: "main_agent",
                    inputTokens: usage.promptTokens,
                    outputTokens: usage.responseTokens,
                    hasSubagents: false,
                  }
                : {
                    usageStatus: "unavailable",
                    usageScope: "main_agent",
                    hasSubagents: false,
                  };
              if (Exit.isFailure(exit) && turnError) {
                context.session = { ...context.session, status: "error", lastError: turnError.detail, updatedAt: yield* nowIso };
                yield* emit({
                  ...(yield* makeEventStamp()),
                  provider: PROVIDER,
                  providerInstanceId: boundInstanceId,
                  threadId: input.threadId,
                  turnId,
                  type: "turn.completed",
                  payload: { state: "failed", errorMessage: turnError.detail, tokenUsage },
                });
                yield* emit({
                  ...(yield* makeEventStamp()),
                  provider: PROVIDER,
                  providerInstanceId: boundInstanceId,
                  threadId: input.threadId,
                  type: "runtime.error",
                  payload: { message: turnError.detail, class: "provider_error" },
                });
                yield* logNative(input.threadId, "ollama/turn/failed", { turnId, error: turnError.detail });
              } else {
                // Failure without turnError means the fiber was interrupted.
                const isStopped = context.stopped.current;
                context.session = { ...context.session, status: "ready", updatedAt: yield* nowIso };
                yield* emit({
                  ...(yield* makeEventStamp()),
                  provider: PROVIDER,
                  providerInstanceId: boundInstanceId,
                  threadId: input.threadId,
                  turnId,
                  type: "turn.completed",
                  payload: { state: isStopped || Exit.isFailure(exit) ? "cancelled" : "completed", tokenUsage },
                });
                yield* logNative(input.threadId, "ollama/turn/completed", { turnId, state: isStopped ? "cancelled" : "completed" });
              }
            }),
          ),
        );

        const fiber = runFork(runTurnLoop);
        context.activeFiber = fiber;
        return { threadId: input.threadId, turnId } satisfies ProviderTurnStartResult;
      },
    );

    const interruptTurn: OllamaAdapterShape["interruptTurn"] = Effect.fn("interruptTurn")(
      function* (threadId: ThreadId) {
        const context = sessions.get(threadId);
        if (context) {
          context.stopped.current = true;
          for (const [, pending] of context.pendingApprovals) {
            yield* Deferred.succeed(pending.decision, "cancel");
          }
          context.pendingApprovals.clear();
          // Interrupt the running fiber and await its termination. The fiber's
          // onExit clears activeTurnId/activeFiber and emits turn.completed.
          if (context.activeFiber) {
            yield* Fiber.interrupt(context.activeFiber);
            context.activeFiber = undefined;
          }
        }
      },
    );

    const respondToRequest: OllamaAdapterShape["respondToRequest"] = Effect.fn("respondToRequest")(
      function* (threadId: ThreadId, requestId: ApprovalRequestId, decision: ProviderApprovalDecision) {
        const context = sessions.get(threadId);
        if (!context) {
          return yield* new ProviderAdapterSessionNotFoundError({ provider: PROVIDER, threadId });
        }
        const pending = context.pendingApprovals.get(requestId);
        if (!pending) {
          return yield* new ProviderAdapterRequestError({
            provider: PROVIDER,
            method: "respondToRequest",
            detail: `Unknown pending approval request: ${requestId}`,
          });
        }
        context.pendingApprovals.delete(requestId);
        yield* Deferred.succeed(pending.decision, decision);
      },
    );

    const respondToUserInput: OllamaAdapterShape["respondToUserInput"] = Effect.fn("respondToUserInput")(function* () {});

    const stopSession: OllamaAdapterShape["stopSession"] = Effect.fn("stopSession")(
      function* (threadId: ThreadId) {
        const context = sessions.get(threadId);
        if (context) {
          context.stopped.current = true;
          for (const [, pending] of context.pendingApprovals) {
            yield* Deferred.succeed(pending.decision, "cancel");
          }
          context.pendingApprovals.clear();
          if (context.activeFiber) {
            yield* Fiber.interrupt(context.activeFiber);
            context.activeFiber = undefined;
          }
          sessions.delete(threadId);
        }
      },
    );

    const listSessions: OllamaAdapterShape["listSessions"] = Effect.fn("listSessions")(
      function* () { return Array.from(sessions.values()).map((ctx) => ctx.session); },
    );

    const hasSession: OllamaAdapterShape["hasSession"] = Effect.fn("hasSession")(
      function* (threadId: ThreadId) { return sessions.has(threadId); },
    );

    const readThread: OllamaAdapterShape["readThread"] = Effect.fn("readThread")(
      function* (threadId: ThreadId) {
        if (!sessions.has(threadId)) {
          return yield* new ProviderAdapterSessionNotFoundError({ provider: PROVIDER, threadId });
        }
        return { threadId, turns: [] };
      },
    );

    const rollbackThread: OllamaAdapterShape["rollbackThread"] = Effect.fn("rollbackThread")(
      function* (threadId: ThreadId, numTurns: number) {
        const context = sessions.get(threadId);
        if (!context) {
          return yield* new ProviderAdapterSessionNotFoundError({ provider: PROVIDER, threadId });
        }
        // turnMessageIndices holds the message-array offset at the start of
        // each turn, so a rollback splices whole turns (incl. tool messages).
        const indices = context.turnMessageIndices;
        if (numTurns <= 0 || indices.length === 0) return { threadId, turns: [] };
        const keepTurns = Math.max(0, indices.length - numTurns);
        const rollbackIndex = indices[keepTurns] ?? indices[0] ?? 0;
        context.messages.splice(rollbackIndex);
        context.turnMessageIndices.splice(keepTurns);
        return { threadId, turns: [] };
      },
    );

    const stopAll: OllamaAdapterShape["stopAll"] = Effect.fn("stopAll")(function* () {
      const keys = Array.from(sessions.keys());
      for (const threadId of keys) {
        yield* stopSession(threadId);
      }
    });

    return {
      provider: PROVIDER,
      capabilities: { sessionModelSwitch: "in-session", supportsConversationRollback: true },
      startSession,
      sendTurn,
      interruptTurn,
      respondToRequest,
      respondToUserInput,
      stopSession,
      listSessions,
      hasSession,
      readThread,
      rollbackThread,
      stopAll,
      streamEvents: Stream.fromQueue(runtimeEvents),
    } satisfies OllamaAdapterShape;
  });