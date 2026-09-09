import * as Effect from "effect/Effect";
import * as Schema from "effect/Schema";
import { HttpClient } from "effect/unstable/http";
import { TextGenerationError, type ModelSelection, type OllamaSettings } from "@t3tools/contracts";
import { extractJsonObject } from "@t3tools/shared/schemaJson";
import { sanitizeBranchFragment, sanitizeFeatureBranchName } from "@t3tools/shared/git";
import { buildBranchNamePrompt, buildCommitMessagePrompt, buildPrContentPrompt, buildThreadTitlePrompt } from "./TextGenerationPrompts.ts";
import * as TextGeneration from "./TextGeneration.ts";
import { sanitizeCommitSubject, sanitizePrTitle, sanitizeThreadTitle } from "./TextGenerationUtils.ts";
import { ollamaChat } from "../provider/ollamaRuntime.js";

/** Fallback text-generation model. Kept in sync with
 * `DEFAULT_TEXT_GENERATION_MODEL_BY_PROVIDER["ollama"]` in contracts. */
const FALLBACK_TEXT_GENERATION_MODEL = "qwen2.5:7b";

export const makeOllamaTextGeneration = Effect.fn("makeOllamaTextGeneration")(function* (
  ollamaSettings: OllamaSettings,
  processEnv?: Record<string, string | undefined>,
) {
  const apiKey = processEnv?.OLLAMA_API_KEY;
  const client = yield* HttpClient.HttpClient;
  const resolveModel = (modelSelection: ModelSelection): string =>
    modelSelection.model?.trim() || ollamaSettings.model?.trim() || FALLBACK_TEXT_GENERATION_MODEL;

  const runOllamaJson = <S extends Schema.Top>(input: {
    readonly operation: string;
    readonly cwd: string;
    readonly prompt: string;
    readonly outputSchemaJson: S;
    readonly modelSelection: ModelSelection;
  }) =>
    Effect.gen(function* () {
      const model = resolveModel(input.modelSelection);
      const response = yield* ollamaChat({ client, baseUrl: ollamaSettings.baseUrl, apiKey, model, messages: [{ role: "user", content: input.prompt }] });
      const rawText = response.message.content.trim();
      if (rawText.length === 0) {
        return yield* new TextGenerationError({ operation: input.operation, detail: "Ollama returned empty output." });
      }
      const decodeOutput = Schema.decodeEffect(Schema.fromJsonString(input.outputSchemaJson));
      return yield* decodeOutput(extractJsonObject(rawText)).pipe(
        Effect.catchTag("SchemaError", (cause) =>
          Effect.fail(new TextGenerationError({ operation: input.operation, detail: "Ollama returned invalid structured output.", cause })),
        ),
      );
    }).pipe(
      // Only the ollamaChat call produces a foreign error; the empty-output and
      // schema-decode paths already fail with TextGenerationError, so catching
      // by tag avoids double-wrapping them.
      Effect.catchTag("OllamaRuntimeError", (cause) =>
        Effect.fail(new TextGenerationError({ operation: input.operation, detail: cause.detail, cause })),
      ),
    );

  const generateCommitMessage: TextGeneration.TextGeneration["Service"]["generateCommitMessage"] =
    Effect.fn("OllamaTextGeneration.generateCommitMessage")(function* (input) {
      const { prompt, outputSchema } = buildCommitMessagePrompt({ branch: input.branch, stagedSummary: input.stagedSummary, stagedPatch: input.stagedPatch, includeBranch: input.includeBranch === true });
      const generated = yield* runOllamaJson({ operation: "generateCommitMessage", cwd: input.cwd, prompt, outputSchemaJson: outputSchema, modelSelection: input.modelSelection });
      return {
        subject: sanitizeCommitSubject(generated.subject),
        body: generated.body.trim(),
        ...("branch" in generated && typeof generated.branch === "string" ? { branch: sanitizeFeatureBranchName(generated.branch) } : {}),
      } satisfies TextGeneration.CommitMessageGenerationResult;
    });

  const generatePrContent: TextGeneration.TextGeneration["Service"]["generatePrContent"] =
    Effect.fn("OllamaTextGeneration.generatePrContent")(function* (input) {
      const { prompt, outputSchema } = buildPrContentPrompt({ baseBranch: input.baseBranch, headBranch: input.headBranch, commitSummary: input.commitSummary, diffSummary: input.diffSummary, diffPatch: input.diffPatch });
      const generated = yield* runOllamaJson({ operation: "generatePrContent", cwd: input.cwd, prompt, outputSchemaJson: outputSchema, modelSelection: input.modelSelection });
      return {
        title: sanitizePrTitle(generated.title),
        body: generated.body.trim(),
      } satisfies TextGeneration.PrContentGenerationResult;
    });

  const generateBranchName: TextGeneration.TextGeneration["Service"]["generateBranchName"] =
    Effect.fn("OllamaTextGeneration.generateBranchName")(function* (input) {
      const { prompt, outputSchema } = buildBranchNamePrompt({ message: input.message, attachments: input.attachments });
      const generated = yield* runOllamaJson({ operation: "generateBranchName", cwd: input.cwd, prompt, outputSchemaJson: outputSchema, modelSelection: input.modelSelection });
      return {
        branch: sanitizeBranchFragment(generated.branch),
      } satisfies TextGeneration.BranchNameGenerationResult;
    });

  const generateThreadTitle: TextGeneration.TextGeneration["Service"]["generateThreadTitle"] =
    Effect.fn("OllamaTextGeneration.generateThreadTitle")(function* (input) {
      const { prompt, outputSchema } = buildThreadTitlePrompt({ message: input.message, previousTitle: input.previousTitle, attachments: input.attachments });
      const generated = yield* runOllamaJson({ operation: "generateThreadTitle", cwd: input.cwd, prompt, outputSchemaJson: outputSchema, modelSelection: input.modelSelection });
      return {
        title: sanitizeThreadTitle(generated.title),
      } satisfies TextGeneration.ThreadTitleGenerationResult;
    });

  return {
    generateCommitMessage,
    generatePrContent,
    generateBranchName,
    generateThreadTitle,
  } satisfies TextGeneration.TextGeneration["Service"];
});