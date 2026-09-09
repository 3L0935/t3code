import { OllamaSettings, ProviderDriverKind } from "@t3tools/contracts";
import * as Crypto from "effect/Crypto";
import * as Effect from "effect/Effect";
import * as Schema from "effect/Schema";
import { HttpClient } from "effect/unstable/http";

import * as BackgroundPolicy from "../../background/BackgroundPolicy.ts";
import { ServerSettingsService } from "../../serverSettings.ts";
import { makeOllamaTextGeneration } from "../../textGeneration/OllamaTextGeneration.ts";
import { ProviderDriverError } from "../Errors.ts";
import { makeOllamaAdapter } from "../Layers/OllamaAdapter.ts";
import { checkOllamaProviderStatus, makePendingOllamaProvider } from "../Layers/OllamaProvider.ts";
import { ProviderEventLoggers } from "../Layers/ProviderEventLoggers.ts";
import { makeManagedServerProvider } from "../makeManagedServerProvider.ts";
import {
  defaultProviderContinuationIdentity,
  type ProviderDriver,
  type ProviderInstance,
} from "../ProviderDriver.ts";
import { withInstanceIdentity } from "./instanceIdentity.ts";
import { mergeProviderInstanceEnvironment } from "../ProviderInstanceEnvironment.ts";
import { makeManualOnlyProviderMaintenanceCapabilities } from "../providerMaintenance.ts";
import {
  haveProviderSnapshotSettingsChanged,
  makeProviderSnapshotSettingsSource,
  type ProviderSnapshotSettings,
} from "../providerUpdateSettings.ts";

const decodeOllamaSettings = Schema.decodeSync(OllamaSettings);

const DRIVER_KIND = ProviderDriverKind.make("ollama");
const MAINTENANCE_CAPABILITIES = makeManualOnlyProviderMaintenanceCapabilities({
  provider: DRIVER_KIND,
  packageName: null,
});

export type OllamaDriverEnv =
  | BackgroundPolicy.BackgroundPolicy
  | Crypto.Crypto
  | HttpClient.HttpClient
  | ProviderEventLoggers
  | ServerSettingsService;

export const OllamaDriver: ProviderDriver<OllamaSettings, OllamaDriverEnv> = {
  driverKind: DRIVER_KIND,
  metadata: { displayName: "Ollama", supportsMultipleInstances: true },
  configSchema: OllamaSettings,
  defaultConfig: (): OllamaSettings => decodeOllamaSettings({}),
  create: ({ instanceId, displayName, accentColor, environment, enabled, config }) =>
    Effect.gen(function* () {
      const serverSettings = yield* ServerSettingsService;
      const httpClient = yield* HttpClient.HttpClient;
      const eventLoggers = yield* ProviderEventLoggers;
      const processEnv = mergeProviderInstanceEnvironment(environment);
      const continuationIdentity = defaultProviderContinuationIdentity({
        driverKind: DRIVER_KIND,
        instanceId,
      });
      const stampIdentity = withInstanceIdentity({
        instanceId,
        driverKind: DRIVER_KIND,
        displayName,
        accentColor,
        continuationGroupKey: continuationIdentity.continuationKey,
      });
      const effectiveConfig = { ...config, enabled } satisfies OllamaSettings;
      const adapter = yield* makeOllamaAdapter(effectiveConfig, processEnv, {
        instanceId,
        ...(eventLoggers.native ? { nativeEventLogger: eventLoggers.native } : {}),
      });
      const textGeneration = yield* makeOllamaTextGeneration(effectiveConfig, processEnv);

      const checkProvider = checkOllamaProviderStatus(effectiveConfig, processEnv).pipe(
        Effect.provideService(HttpClient.HttpClient, httpClient),
        Effect.map(stampIdentity),
      );

      const snapshotSettings = makeProviderSnapshotSettingsSource(effectiveConfig, serverSettings);
      const snapshot = yield* makeManagedServerProvider<ProviderSnapshotSettings<OllamaSettings>>({
        resolveMaintenance: () => Effect.succeed(MAINTENANCE_CAPABILITIES),
        getSettings: snapshotSettings.getSettings,
        streamSettings: snapshotSettings.streamSettings,
        haveSettingsChanged: haveProviderSnapshotSettingsChanged,
        initialSnapshot: (settings) =>
          makePendingOllamaProvider(settings.provider).pipe(Effect.map(stampIdentity)),
        checkProvider,
      }).pipe(
        Effect.mapError(
          (cause) =>
            new ProviderDriverError({
              driver: DRIVER_KIND,
              instanceId,
              detail: `Failed to build Ollama snapshot: ${cause.message ?? String(cause)}`,
              cause,
            }),
        ),
      );

      return {
        instanceId,
        driverKind: DRIVER_KIND,
        continuationIdentity,
        displayName,
        accentColor,
        enabled,
        snapshot,
        adapter,
        textGeneration,
      } satisfies ProviderInstance;
    }),
};