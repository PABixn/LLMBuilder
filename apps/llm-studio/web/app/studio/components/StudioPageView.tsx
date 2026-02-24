import type {
  ChangeEvent,
  Dispatch,
  DragEvent,
  RefObject,
  SetStateAction,
} from "react";
import {
  FiAlertTriangle,
  FiCheckCircle,
  FiCopy,
  FiDownload,
  FiHardDrive,
  FiLayers,
  FiMoon,
  FiPlus,
  FiRefreshCw,
  FiServer,
  FiSun,
  FiUpload,
  FiXCircle,
} from "react-icons/fi";

import { apiBaseUrl } from "../../../lib/api";
import type { ModelConfig } from "../../../lib/defaults";

import { BuilderPanel, type BuilderPanelProps } from "./builder/BuilderPanel";
import { PaletteTile, StatusCard } from "./primitives";
import type {
  BackendAnalysisState,
  BackendValidationState,
  BuilderMetrics,
  Diagnostic,
  NoticeState,
  StudioComponentKind,
  StudioDocument,
  ThemeMode,
} from "../types";
import { labelForComponentKind } from "../utils/document";
import { formatBytes, formatCompactCount, formatTimeAgo, integerInputValue, parseIntegerInput } from "../utils/format";

export interface StudioPageViewProps extends BuilderPanelProps {
  fileInputRef: RefObject<HTMLInputElement | null>;
  importFromFile: (event: ChangeEvent<HTMLInputElement>) => Promise<void>;
  theme: ThemeMode;
  setTheme: Dispatch<SetStateAction<ThemeMode>>;
  addBlock: () => void;
  exportJson: () => void;
  resetDefaults: () => void;
  notice: NoticeState | null;
  validationStatusLabel: string;
  backendValidation: BackendValidationState;
  totalErrors: number;
  totalWarnings: number;
  metrics: BuilderMetrics;
  lastSavedAt: number | null;
  documentState: StudioDocument;
  updateBaseField: (
    key: keyof Pick<StudioDocument, "context_length" | "vocab_size" | "n_embd">,
    value: number
  ) => void;
  setDocumentState: Dispatch<SetStateAction<StudioDocument>>;
  beginDragPaletteComponent: (event: DragEvent<HTMLDivElement>, componentKind: StudioComponentKind) => void;
  clearDragState: () => void;
  diagnostics: Diagnostic[];
  localErrors: Diagnostic[];
  localWarnings: Diagnostic[];
  backendAnalysis: BackendAnalysisState;
  backendAnalysisStale: boolean;
  runBackendAnalysis: () => Promise<void>;
  previewJson: string;
  copyJson: () => Promise<void>;
  importDraft: string;
  setImportDraft: Dispatch<SetStateAction<string>>;
  applyImportText: (text: string) => void;
  modelConfig: ModelConfig;
}

export function StudioPageView({
  fileInputRef,
  importFromFile,
  theme,
  setTheme,
  addBlock,
  exportJson,
  resetDefaults,
  notice,
  validationStatusLabel,
  backendValidation,
  totalErrors,
  totalWarnings,
  metrics,
  lastSavedAt,
  documentState,
  updateBaseField,
  setDocumentState,
  beginDragPaletteComponent,
  clearDragState,
  diagnostics,
  localErrors,
  localWarnings,
  backendAnalysis,
  backendAnalysisStale,
  runBackendAnalysis,
  previewJson,
  copyJson,
  importDraft,
  setImportDraft,
  applyImportText,
  modelConfig,
  ...builderPanelProps
}: StudioPageViewProps) {
  const localActivationTotal =
    metrics.activationCount + metrics.mlpActivationStepCount;
  const backendActivationTotal =
    (backendAnalysis.summary?.activation_component_count ?? 0) +
    (backendAnalysis.summary?.mlp_activation_step_count ?? 0);
  const moduleInventoryEntries = backendAnalysis.summary
    ? Object.entries(backendAnalysis.summary.module_counts).sort(
        ([nameA, countA], [nameB, countB]) =>
          countB - countA || nameA.localeCompare(nameB)
      )
    : [];
  const visibleModuleInventoryEntries = moduleInventoryEntries.slice(0, 8);
  const hiddenModuleInventoryCount = Math.max(
    0,
    moduleInventoryEntries.length - visibleModuleInventoryEntries.length
  );

  return (
    <main className="studioRoot">
      <nav className="studioNav" aria-label="LLM Studio navigation">
        <div className="studioNavBrand">
          <span className="studioNavDot" />
          <span>LLM Studio</span>
        </div>
        <div className="studioNavLinks">
          <a className="studioNavLink" href="#base-model">
            Base Model
          </a>
          <a className="studioNavLink" href="#block-builder">
            Builder
          </a>
          <a className="studioNavLink" href="#diagnostics">
            Diagnostics
          </a>
          <a className="studioNavLink" href="#model-analysis">
            Analysis
          </a>
          <a className="studioNavLink" href="#json-preview">
            JSON
          </a>
        </div>
        <button
          type="button"
          className="themeToggle"
          onClick={() => setTheme((current) => (current === "dark" ? "white" : "dark"))}
          aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
          title={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
        >
          {theme === "dark" ? <FiSun /> : <FiMoon />}
        </button>
      </nav>

      <section className="panelCard heroCard">
        <div className="panelHead heroHead">
          <div>
            <p className="panelEyebrow">Visual /model Designer</p>
            <h1>Compose transformer blocks, validate live, export clean JSON.</h1>
            <p className="panelCopy">
              Build `{"/model"}` configs using draggable block components and nested MLP step editors.
              LLM Studio keeps a local semantic validator active even when the backend validator is offline.
            </p>
          </div>
          <div className="heroActions">
            <button type="button" className="buttonGhost" onClick={addBlock}>
              <FiPlus /> Add Block
            </button>
            <button type="button" className="buttonGhost" onClick={() => fileInputRef.current?.click()}>
              <FiUpload /> Import File
            </button>
            <button type="button" className="buttonGhost" onClick={exportJson}>
              <FiDownload /> Export JSON
            </button>
            <button type="button" className="buttonGhost" onClick={resetDefaults}>
              <FiRefreshCw /> Reset
            </button>
          </div>
        </div>

        {notice ? (
          <div className={`inlineNotice tone-${notice.tone}`} role="status" aria-live="polite">
            {notice.message}
          </div>
        ) : null}

        <div className="statusGrid">
          <StatusCard
            title="Validation"
            value={validationStatusLabel}
            detail={backendValidation.message}
            tone={
              totalErrors > 0
                ? "bad"
                : totalWarnings > 0 || backendValidation.phase === "fallback"
                  ? "warn"
                  : "good"
            }
            icon={totalErrors > 0 ? <FiXCircle /> : totalWarnings > 0 ? <FiAlertTriangle /> : <FiCheckCircle />}
          />
          <StatusCard
            title="Blocks"
            value={`${metrics.blockCount}`}
            detail={`${metrics.componentCount} components · ${metrics.mlpStepCount} MLP steps`}
            tone="neutral"
            icon={<FiLayers />}
          />
          <StatusCard
            title="Backend"
            value={backendValidation.phase === "success" ? "Connected" : backendValidation.phase === "validating" ? "Validating" : backendValidation.phase === "fallback" ? "Fallback" : "Idle"}
            detail={`${apiBaseUrl()} · ${formatTimeAgo(backendValidation.lastValidatedAt)}`}
            tone={backendValidation.phase === "success" ? "good" : backendValidation.phase === "fallback" ? "warn" : "neutral"}
            icon={<FiServer />}
          />
          <StatusCard
            title="Local Cache"
            value="Auto-saved"
            detail={`Last save ${formatTimeAgo(lastSavedAt)}`}
            tone="neutral"
            icon={<FiHardDrive />}
          />
        </div>
      </section>

      <div className="twoColLayout">
        <section id="base-model" className="panelCard">
          <div className="panelHead">
            <div>
              <p className="panelEyebrow">Base Model</p>
              <h2>Core config controls</h2>
              <p className="panelCopy">
                These values shape the shared model dimensions and drive semantic checks for attention head sizing.
              </p>
            </div>
          </div>
          <div className="fieldGrid">
            <label className="fieldLabel" htmlFor="context_length">
              <span>context_length</span>
              <input
                id="context_length"
                type="number"
                min={1}
                step={1}
                value={integerInputValue(documentState.context_length)}
                onChange={(event) =>
                  updateBaseField(
                    "context_length",
                    parseIntegerInput(event.target.value, documentState.context_length)
                  )
                }
              />
            </label>
            <label className="fieldLabel" htmlFor="vocab_size">
              <span>vocab_size</span>
              <input
                id="vocab_size"
                type="number"
                min={1}
                step={1}
                value={integerInputValue(documentState.vocab_size)}
                onChange={(event) =>
                  updateBaseField(
                    "vocab_size",
                    parseIntegerInput(event.target.value, documentState.vocab_size)
                  )
                }
              />
            </label>
            <label className="fieldLabel" htmlFor="n_embd">
              <span>n_embd</span>
              <input
                id="n_embd"
                type="number"
                min={1}
                step={1}
                value={integerInputValue(documentState.n_embd)}
                onChange={(event) =>
                  updateBaseField("n_embd", parseIntegerInput(event.target.value, documentState.n_embd))
                }
              />
            </label>
            <label className="toggleField tall" htmlFor="weight_tying">
              <input
                id="weight_tying"
                type="checkbox"
                checked={documentState.weight_tying}
                onChange={(event) =>
                  setDocumentState((current) => ({ ...current, weight_tying: event.target.checked }))
                }
              />
              <span>weight_tying</span>
            </label>
          </div>
        </section>

        <section className="panelCard">
          <div className="panelHead">
            <div>
              <p className="panelEyebrow">Palette</p>
              <h2>Drag components into blocks</h2>
              <p className="panelCopy">
                Use native drag-and-drop to attach components to any block row or reorder existing components.
              </p>
            </div>
          </div>
          <div className="paletteGrid">
            {([
              {
                kind: "attention",
                subtitle: "Self-attention with n_head / n_kv_head",
                colorClass: "tone-attention",
              },
              {
                kind: "mlp",
                subtitle: "Configurable MLP with nested sequence editor",
                colorClass: "tone-mlp",
              },
              {
                kind: "norm",
                subtitle: "LayerNorm or RMSNorm (optional learnable gamma)",
                colorClass: "tone-norm",
              },
              {
                kind: "activation",
                subtitle: "Standalone activation block component",
                colorClass: "tone-activation",
              },
            ] as const).map((entry) => (
              <PaletteTile
                key={entry.kind}
                title={labelForComponentKind(entry.kind)}
                subtitle={entry.subtitle}
                colorClass={entry.colorClass}
                draggable
                onDragStart={(event) => beginDragPaletteComponent(event, entry.kind)}
                onDragEnd={clearDragState}
              />
            ))}
          </div>
        </section>
      </div>


      <BuilderPanel
        {...builderPanelProps}
        documentState={documentState}
        metrics={metrics}
        addBlock={addBlock}
        clearDragState={clearDragState}
      />

      <section id="diagnostics" className="panelCard diagnosticsPanel">
        <div className="panelHead">
          <div>
            <p className="panelEyebrow">Diagnostics</p>
            <h2>Validation and warnings</h2>
            <p className="panelCopy">
              Local semantic checks run on every edit. Backend `{"/validate/model"}` validation is attempted when local errors are clear.
            </p>
          </div>
        </div>

        <div className="diagnosticSummaryRow">
          <div className="pillBadge tone-error">{localErrors.length} local errors</div>
          <div className="pillBadge tone-warn">{localWarnings.length} local warnings</div>
          <div className="pillBadge tone-error">{backendValidation.errors.length} backend errors</div>
          <div className="pillBadge tone-warn">{backendValidation.warnings.length} backend warnings</div>
          <div className={`pillBadge ${backendValidation.phase === "success" ? "tone-good" : backendValidation.phase === "fallback" ? "tone-warn" : "tone-neutral"}`}>
            backend: {backendValidation.phase}
          </div>
        </div>

        <div className="diagnosticList" role="list">
          {diagnostics.length === 0 ? (
            <div className="diagnosticItem tone-good" role="listitem">
              <div className="diagnosticIcon">
                <FiCheckCircle />
              </div>
              <div>
                <div className="diagnosticTitle">No local or backend warnings.</div>
                <div className="diagnosticMeta">Configuration looks ready to export.</div>
              </div>
            </div>
          ) : (
            diagnostics.map((diagnostic) => (
              <div
                key={diagnostic.id}
                className={`diagnosticItem tone-${diagnostic.level}`}
                role="listitem"
              >
                <div className="diagnosticIcon">
                  {diagnostic.level === "error" ? (
                    <FiXCircle />
                  ) : diagnostic.level === "warning" ? (
                    <FiAlertTriangle />
                  ) : (
                    <FiCheckCircle />
                  )}
                </div>
                <div>
                  <div className="diagnosticTitle">{diagnostic.message}</div>
                  <div className="diagnosticMeta">
                    <code>{diagnostic.path}</code> · {diagnostic.source}
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </section>

      <section id="model-analysis" className="panelCard analysisPanel">
        <div className="panelHead">
          <div>
            <p className="panelEyebrow">Backend Model Analysis</p>
            <h2>Instantiate `ConfigurableGPT` and inspect runtime-facing metrics</h2>
            <p className="panelCopy">
              Runs on the backend using `{"/model/model.py"}` to confirm the config constructs a real model and to estimate parameter and KV-cache memory.
            </p>
          </div>
          <div className="actionCluster">
            <button
              type="button"
              className="buttonGhost"
              onClick={() => {
                void runBackendAnalysis();
              }}
              disabled={backendAnalysis.phase === "running" || localErrors.length > 0}
            >
              <FiServer /> {backendAnalysis.phase === "running" ? "Analyzing..." : "Run Analysis"}
            </button>
          </div>
        </div>

        <div className="diagnosticSummaryRow">
          <div
            className={`pillBadge ${
              backendAnalysis.phase === "success"
                ? "tone-good"
                : backendAnalysis.phase === "error"
                  ? "tone-error"
                  : backendAnalysis.phase === "running"
                    ? "tone-warn"
                    : "tone-neutral"
            }`}
          >
            analysis: {backendAnalysis.phase}
          </div>
          <div className="pillBadge tone-neutral">
            last run: {formatTimeAgo(backendAnalysis.lastAnalyzedAt)}
          </div>
          {backendAnalysisStale ? (
            <div className="pillBadge tone-warn">stale vs current draft</div>
          ) : null}
        </div>

        <p className="analysisMessage">{backendAnalysis.message}</p>

        {backendAnalysis.summary ? (
          <>
            <div className="statusGrid analysisStatsGrid">
              <StatusCard
                title="Parameters"
                value={formatCompactCount(backendAnalysis.summary.total_parameters)}
                detail={`${formatBytes(backendAnalysis.summary.parameter_memory_bytes_fp32)} fp32 · ${formatBytes(backendAnalysis.summary.parameter_memory_bytes_bf16)} bf16`}
                tone="good"
                icon={<FiLayers />}
              />
              <StatusCard
                title="KV Cache / Token"
                value={formatBytes(
                  backendAnalysis.summary.estimated_kv_cache_bytes_per_token_fp16
                )}
                detail={`${formatBytes(backendAnalysis.summary.estimated_kv_cache_bytes_for_context_fp16)} @ context_length`}
                tone="neutral"
                icon={<FiHardDrive />}
              />
              <StatusCard
                title="Head Dim"
                value={
                  backendAnalysis.summary.min_head_dim === null
                    ? "n/a"
                    : backendAnalysis.summary.min_head_dim ===
                        backendAnalysis.summary.max_head_dim
                      ? `${backendAnalysis.summary.min_head_dim}`
                      : `${backendAnalysis.summary.min_head_dim}-${backendAnalysis.summary.max_head_dim}`
                }
                detail={`${backendAnalysis.summary.attention_component_count} attention components`}
                tone="neutral"
                icon={<FiServer />}
              />
              <StatusCard
                title="Instantiation"
                value={`${backendAnalysis.summary.instantiation_time_ms.toFixed(1)} ms`}
                detail={`${formatCompactCount(backendAnalysis.summary.trainable_parameters)} trainable`}
                tone="neutral"
                icon={<FiRefreshCw />}
              />
            </div>

            <div className="twoColLayout analysisLayout">
              <div className="workflowItem">
                <div className="workflowTitle">Component counts</div>
                <div className="analysisChipRow">
                  <span>{backendAnalysis.summary.block_count} blocks</span>
                  <span>{backendAnalysis.summary.component_count} components</span>
                  <span>{backendAnalysis.summary.attention_component_count} attention</span>
                  <span>{backendAnalysis.summary.mlp_component_count} mlp</span>
                  <span>{backendAnalysis.summary.norm_component_count} norm</span>
                  <span>{backendActivationTotal} activations</span>
                </div>
              </div>

              <div className="workflowItem">
                <div className="workflowTitle">Module inventory</div>
                <div className="analysisChipRow analysisModuleChipRow">
                  {visibleModuleInventoryEntries.map(([name, count]) => (
                    <span key={name} className="analysisModuleChip">
                      <code>{name}</code>
                      <strong>{count}</strong>
                    </span>
                  ))}
                  {hiddenModuleInventoryCount > 0 ? (
                    <span className="analysisModuleChip isMeta">
                      +{hiddenModuleInventoryCount} more
                    </span>
                  ) : null}
                </div>
              </div>
            </div>
          </>
        ) : null}

        {backendAnalysis.instantiationError ? (
          <div className="diagnosticList">
            <div className="diagnosticItem tone-error" role="listitem">
              <div className="diagnosticIcon">
                <FiXCircle />
              </div>
              <div>
                <div className="diagnosticTitle">Model instantiation failed</div>
                <div className="diagnosticMeta">{backendAnalysis.instantiationError}</div>
              </div>
            </div>
          </div>
        ) : null}
      </section>

      <div id="json-preview" className="twoColLayout previewLayout">
        <section className="panelCard previewPanel">
          <div className="panelHead">
            <div>
              <p className="panelEyebrow">JSON Preview</p>
              <h2>Exportable model config</h2>
              <p className="panelCopy">
                Live JSON mirrors the visual state. Use copy/export for downstream training or backend validation calls.
              </p>
            </div>
            <div className="actionCluster">
              <button type="button" className="buttonGhost" onClick={copyJson}>
                <FiCopy /> Copy
              </button>
              <button type="button" className="buttonGhost" onClick={exportJson}>
                <FiDownload /> Export
              </button>
            </div>
          </div>
          <pre className="jsonPreview"><code>{previewJson}</code></pre>
        </section>

        <section className="panelCard importPanel">
          <div className="panelHead">
            <div>
              <p className="panelEyebrow">Import / Workflow</p>
              <h2>Paste or load JSON</h2>
              <p className="panelCopy">
                Import a `/model` JSON document to regenerate blocks and nested MLP sequences. Unsupported shapes are rejected before state changes.
              </p>
            </div>
          </div>

          <div className="actionRowWrap">
            <button type="button" className="buttonGhost" onClick={() => fileInputRef.current?.click()}>
              <FiUpload /> Choose JSON File
            </button>
            <button type="button" className="buttonGhost" onClick={() => applyImportText(importDraft)}>
              <FiRefreshCw /> Apply Import Text
            </button>
            <button
              type="button"
              className="buttonGhost"
              onClick={() => setImportDraft(JSON.stringify(modelConfig, null, 2))}
            >
              <FiCopy /> Load Current Into Editor
            </button>
          </div>

          <label className="fieldLabel" htmlFor="import-draft">
            <span>Import JSON</span>
            <textarea
              id="import-draft"
              value={importDraft}
              onChange={(event) => setImportDraft(event.target.value)}
              placeholder="Paste /model JSON here..."
              rows={16}
            />
          </label>

          <div className="workflowList">
            <div className="workflowItem">
              <div className="workflowTitle">Suggested workflow</div>
              <ol>
                <li>Set base dimensions (`n_embd`, `context_length`, `vocab_size`).</li>
                <li>Compose one reference block, then duplicate and tune variants.</li>
                <li>Resolve diagnostics, then export JSON or submit to backend validator.</li>
              </ol>
            </div>
            <div className="workflowItem">
              <div className="workflowTitle">Counts</div>
              <div className="workflowStats">
                <span>{metrics.attentionCount} attention</span>
                <span>{metrics.mlpCount} mlp</span>
                <span>{metrics.normCount} norm</span>
                <span>{localActivationTotal} activations</span>
              </div>
            </div>
          </div>
        </section>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="application/json,.json"
        hidden
        onChange={(event) => {
          void importFromFile(event);
        }}
      />
    </main>
  );
}
