import * as DateTime from "effect/DateTime";
import * as Effect from "effect/Effect";
import * as Exit from "effect/Exit";
import * as Cause from "effect/Cause";
import { HttpClient } from "effect/unstable/http";
import { type CustomModelSetting, type OllamaSettings, ProviderDriverKind, type ModelCapabilities } from "@t3tools/contracts";

import { createModelCapabilities } from "@t3tools/shared/model";
import { buildServerProvider, providerModelsFromSettings, type ServerProviderDraft } from "../providerSnapshot.js";
import { ollamaListModels, ollamaVersion } from "../ollamaRuntime.js";

const PROVIDER = ProviderDriverKind.make("ollama");
const OLLAMA_PRESENTATION = { displayName: "Ollama", showInteractionModeToggle: false } as const;
const DEFAULT_CAPABILITIES: ModelCapabilities = createModelCapabilities({ optionDescriptors: [] });

function ollamaModelsFromSettings(
  customModels: ReadonlyArray<CustomModelSetting> | undefined,
  discoveredModels: ReadonlyArray<{ slug: string }> = [],
): ReturnType<typeof providerModelsFromSettings> {
  return providerModelsFromSettings(
    discoveredModels.map((model) => ({ slug: model.slug, name: model.slug, isCustom: false as const, capabilities: DEFAULT_CAPABILITIES })),
    customModels ?? [],
    DEFAULT_CAPABILITIES,
  );
}

export const makePendingOllamaProvider = (
  ollamaSettings: OllamaSettings,
): Effect.Effect<ServerProviderDraft> =>
  Effect.gen(function* () {
    const checkedAt = DateTime.formatIso(yield* DateTime.now);
    const models = ollamaModelsFromSettings(ollamaSettings.customModels);
    if (!ollamaSettings.enabled) {
      return buildServerProvider({ presentation: OLLAMA_PRESENTATION, enabled: false, checkedAt, models, probe: { installed: false, version: null, status: "warning", auth: { status: "unknown" }, message: "Ollama is disabled." } });
    }
    return buildServerProvider({ presentation: OLLAMA_PRESENTATION, enabled: true, checkedAt, models, probe: { installed: false, version: null, status: "warning", auth: { status: "unknown" }, message: "Ollama status has not been checked yet." } });
  });

export const checkOllamaProviderStatus = (ollamaSettings: OllamaSettings, processEnv: Record<string, string | undefined>) =>
  Effect.gen(function* () {
  const checkedAt = DateTime.formatIso(yield* DateTime.now);
  const baseUrl = ollamaSettings.baseUrl;
  const apiKey = processEnv.OLLAMA_API_KEY;
  const customModels = ollamaSettings.customModels;
  const client = yield* HttpClient.HttpClient;

  if (!ollamaSettings.enabled) {
    return buildServerProvider({ presentation: OLLAMA_PRESENTATION, enabled: false, checkedAt, models: ollamaModelsFromSettings(customModels), probe: { installed: false, version: null, status: "warning", auth: { status: "unknown" }, message: "Ollama is disabled." } });
  }

  // Version is non-critical: a failed probe must not crash the whole status
  // check before ollamaListModels (which carries its own error snapshot).
  // Ollama Cloud reports a placeholder "0.0.0" — treat it as no version.
  const versionExit = yield* Effect.exit(ollamaVersion(client, baseUrl, apiKey));
  const rawVersion = Exit.isSuccess(versionExit) ? versionExit.value : "";
  const version = rawVersion === "0.0.0" ? "" : rawVersion;
  const modelsExit = yield* Effect.exit(ollamaListModels(client, baseUrl, apiKey));

  if (Exit.isFailure(modelsExit)) {
    const detail = Cause.isCause(modelsExit.cause) ? Cause.pretty(modelsExit.cause) : String(modelsExit.cause);
    return buildServerProvider({ presentation: OLLAMA_PRESENTATION, enabled: true, checkedAt, models: ollamaModelsFromSettings(customModels), probe: { installed: true, version: version || null, status: "error", auth: { status: "unknown" }, message: `Could not reach Ollama at ${baseUrl}: ${detail}` } });
  }

  const models = ollamaModelsFromSettings(customModels, modelsExit.value.map((m) => ({ slug: m.name })));

  return buildServerProvider({ presentation: OLLAMA_PRESENTATION, enabled: true, checkedAt, models, probe: { installed: true, version: version || null, status: "ready", auth: { status: "authenticated", type: "ollama" }, message: `${models.length} model${models.length === 1 ? "" : "s"} available via Ollama at ${baseUrl}.` } });
});