import type {
  ChangeEvent,
  Dispatch,
  RefObject,
  SetStateAction,
} from "react";

import type { ModelConfig } from "../../../lib/defaults";

import { BaseModelPanel } from "./page/BaseModelPanel";
import { BackendAnalysisPanel } from "./page/BackendAnalysisPanel";
import { DiagnosticsPanel } from "./page/DiagnosticsPanel";
import { HeroSection } from "./page/HeroSection";
import { JsonWorkspacePanels } from "./page/JsonWorkspacePanels";
import { StudioTopNav } from "./page/StudioTopNav";
import { BuilderPanel, type BuilderPanelProps } from "./builder/BuilderPanel";
import type {
  BackendAnalysisState,
  BackendValidationState,
  BuilderMetrics,
  Diagnostic,
  NoticeState,
  StudioDocument,
  ThemeMode,
} from "../types";

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

function maybeSelectZeroNumberInput(target: EventTarget | null): void {
  if (!(target instanceof HTMLInputElement) || target.type !== "number") {
    return;
  }
  if (target.value.trim() === "") {
    return;
  }
  const numericValue = Number(target.value);
  if (!Number.isFinite(numericValue) || numericValue !== 0) {
    return;
  }
  requestAnimationFrame(() => {
    if (document.activeElement === target) {
      target.select();
    }
  });
}

function maybeSelectForcedZeroAfterEmpty(target: EventTarget | null): void {
  if (!(target instanceof HTMLInputElement) || target.type !== "number") {
    return;
  }
  if (target.value !== "") {
    return;
  }
  requestAnimationFrame(() => {
    if (document.activeElement !== target) {
      return;
    }
    if (target.value.trim() !== "") {
      const numericValue = Number(target.value);
      if (Number.isFinite(numericValue) && numericValue === 0) {
        target.select();
      }
    }
  });
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
  return (
    <main
      className="studioRoot"
      onFocusCapture={(event) => maybeSelectZeroNumberInput(event.target)}
      onClickCapture={(event) => maybeSelectZeroNumberInput(event.target)}
      onChangeCapture={(event) => maybeSelectForcedZeroAfterEmpty(event.target)}
    >
      <StudioTopNav theme={theme} setTheme={setTheme} />

      <HeroSection
        fileInputRef={fileInputRef}
        addBlock={addBlock}
        exportJson={exportJson}
        resetDefaults={resetDefaults}
        notice={notice}
        validationStatusLabel={validationStatusLabel}
        backendValidation={backendValidation}
        totalErrors={totalErrors}
        totalWarnings={totalWarnings}
        metrics={metrics}
        lastSavedAt={lastSavedAt}
      />

      <div className="twoColLayout">
        <BaseModelPanel
          documentState={documentState}
          updateBaseField={updateBaseField}
          setDocumentState={setDocumentState}
        />
        <DiagnosticsPanel
          diagnostics={diagnostics}
          localErrors={localErrors}
          localWarnings={localWarnings}
          backendValidation={backendValidation}
          totalErrors={totalErrors}
          totalWarnings={totalWarnings}
          modelConfig={modelConfig}
          metrics={metrics}
          backendAnalysis={backendAnalysis}
          backendAnalysisStale={backendAnalysisStale}
          consecutiveBlockGroups={builderPanelProps.consecutiveBlockGroups}
          setDocumentState={setDocumentState}
          runBackendAnalysis={runBackendAnalysis}
        />
      </div>

      <BuilderPanel
        {...builderPanelProps}
        documentState={documentState}
        metrics={metrics}
        addBlock={addBlock}
        clearDragState={clearDragState}
      />

      <BackendAnalysisPanel
        backendAnalysis={backendAnalysis}
        backendAnalysisStale={backendAnalysisStale}
        localErrorCount={localErrors.length}
        runBackendAnalysis={runBackendAnalysis}
      />

      <JsonWorkspacePanels
        fileInputRef={fileInputRef}
        previewJson={previewJson}
        copyJson={copyJson}
        exportJson={exportJson}
        importDraft={importDraft}
        setImportDraft={setImportDraft}
        applyImportText={applyImportText}
        modelConfig={modelConfig}
        metrics={metrics}
      />

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
