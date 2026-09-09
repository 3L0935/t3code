import { describe, expect, it } from "@effect/vitest";
import * as Effect from "effect/Effect";
import * as Layer from "effect/Layer";
import * as Stream from "effect/Stream";
import { HttpClient, HttpClientResponse, HttpClientRequest } from "effect/unstable/http";

import { ollamaChat, ollamaChatStream, parseChunkLineForTest } from "./ollamaRuntime.js";

const ndjsonResponse = (chunks: ReadonlyArray<unknown>, status = 200) =>
  new Response(chunks.map((chunk) => JSON.stringify(chunk)).join("\n"), {
    status,
    headers: { "content-type": "application/x-ndjson" },
  });

const makeHttpClientLayer = (respond: (request: HttpClientRequest.HttpClientRequest) => Response) =>
  Layer.succeed(
    HttpClient.HttpClient,
    HttpClient.make((request) => Effect.sync(() => HttpClientResponse.fromWeb(request, respond(request)))),
  );

describe("ollamaRuntime", () => {
  describe("parseChunkLineForTest", () => {
    it("maps snake_case counters to camelCase and defaults message fields", () => {
      const parsed = parseChunkLineForTest(
        JSON.stringify({
          model: "qwen2.5:7b",
          created_at: "2026-05-16T00:00:00Z",
          message: { role: "assistant", content: "hi" },
          done: true,
          prompt_eval_count: 11,
          eval_count: 22,
        }),
      );
      expect(parsed.model).toBe("qwen2.5:7b");
      expect(parsed.done).toBe(true);
      expect(parsed.promptEvalCount).toBe(11);
      expect(parsed.evalCount).toBe(22);
    });

    it("preserves tool_calls arrays", () => {
      const parsed = parseChunkLineForTest(
        JSON.stringify({
          model: "q",
          message: {
            role: "assistant",
            content: "",
            tool_calls: [{ function: { name: "bash", arguments: { command: "ls" } } }],
          },
          done: false,
        }),
      );
      expect(parsed.message.tool_calls).toEqual([{ function: { name: "bash", arguments: { command: "ls" } } }]);
    });
  });

  describe("ollamaChatStream", () => {
    it.effect("emits one chunk per NDJSON line with counters mapped", () =>
      Effect.gen(function* () {
        const client = yield* HttpClient.HttpClient;
        const collected = yield* Stream.runCollect(
          ollamaChatStream({
            client,
            baseUrl: "http://localhost:11434",
            model: "q",
            messages: [{ role: "user", content: "hi" }],
          }),
        );
        const chunks = Array.from(collected);
        expect(chunks).toHaveLength(2);
        expect(chunks[0]?.message.content).toBe("he");
        expect(chunks[1]?.message.content).toBe("llo");
        expect(chunks[1]?.promptEvalCount).toBe(5);
        expect(chunks[1]?.evalCount).toBe(7);
      }).pipe(
        Effect.provide(
          makeHttpClientLayer(() =>
            ndjsonResponse([
              { model: "q", message: { role: "assistant", content: "he" }, done: false },
              {
                model: "q",
                message: { role: "assistant", content: "llo" },
                done: true,
                prompt_eval_count: 5,
                eval_count: 7,
              },
            ]),
          ),
        ),
      ),
    );

    it.effect("fails with OllamaRuntimeError on non-2xx status", () =>
      Effect.gen(function* () {
        const client = yield* HttpClient.HttpClient;
        const result = yield* Stream.runCollect(
          ollamaChatStream({
            client,
            baseUrl: "http://localhost:11434",
            model: "missing",
            messages: [{ role: "user", content: "hi" }],
          }),
        ).pipe(Effect.exit);
        expect(result._tag).toBe("Failure");
        if (result._tag === "Failure") {
          const causes = (result.cause as { readonly reasons?: ReadonlyArray<{ readonly error?: { readonly detail?: string } }> })
            .reasons ?? [];
          const detail = causes.map((reason) => reason.error?.detail ?? "").join(" ");
          expect(detail).toContain("404");
        }
      }).pipe(
        Effect.provide(makeHttpClientLayer(() => new Response("model not found", { status: 404 }))),
      ),
    );
  });

  describe("ollamaChat", () => {
    it.effect("maps the final response with counters", () =>
      Effect.gen(function* () {
        const client = yield* HttpClient.HttpClient;
        const response = yield* ollamaChat({
          client,
          baseUrl: "http://localhost:11434",
          model: "q",
          messages: [{ role: "user", content: "hi" }],
        });
        expect(response.message.content).toBe("answer");
        expect(response.done).toBe(true);
        expect(response.promptEvalCount).toBe(9);
        expect(response.evalCount).toBe(4);
      }).pipe(
        Effect.provide(
          makeHttpClientLayer(() =>
            ndjsonResponse([
              {
                model: "q",
                created_at: "2026-05-16T00:00:00Z",
                message: { role: "assistant", content: "answer" },
                done: true,
                prompt_eval_count: 9,
                eval_count: 4,
              },
            ]),
          ),
        ),
      ),
    );
  });
});