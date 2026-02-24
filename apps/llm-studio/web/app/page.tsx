"use client";

import {
  Fragment,
  startTransition,
  useDeferredValue,
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
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

import { analyzeModelConfig, apiBaseUrl, validateModelConfig } from "../lib/api";
import {
  createDefaultBlockConfig,
  createDefaultModelConfig,
  type ModelConfig,
} from "../lib/defaults";

import { StudioPageView } from "./studio/components/StudioPageView";
import {
  DOCUMENT_STORAGE_KEY,
  DND_MIME,
  IMPORT_DRAFT_STORAGE_KEY,
  THEME_STORAGE_KEY,
  VALIDATION_DEBOUNCE_MS,
  type BackendAnalysisState,
  type BackendValidationState,
  type BlockInsertPreset,
  type Diagnostic,
  type DragPayload,
  type MlpStepKind,
  type NoticeState,
  type NoticeTone,
  type StudioComponent,
  type StudioComponentKind,
  type StudioDocument,
  type StudioMlpStep,
  type ThemeMode,
} from "./studio/types";
import {
  clamp,
  clone,
  cloneBlockWithNewIds,
  collectAllComponentIds,
  collectAllMlpStepIds,
  collectBuilderMetrics,
  collectConsecutiveIdenticalBlockGroups,
  createDefaultStudioComponent,
  createDefaultStudioMlpStep,
  findBlockIndex,
  findComponentIndex,
  getMlpComponent,
  labelForComponentKind,
  studioBlockFromConfig,
  studioDocumentFromConfig,
  studioDocumentToConfig,
} from "./studio/utils/document";
import {
  downloadTextFile,
  formatBytes,
  formatCompactCount,
  formatTimeAgo,
  integerInputValue,
  parseIntegerInput,
} from "./studio/utils/format";
import {
  parseImportedModelConfig,
  pushDiagnostic,
  validateLocalConfig,
} from "./studio/utils/validation";

export default function Page() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const dragRef = useRef<DragPayload | null>(null);
  const validationRunRef = useRef(0);

  const [theme, setTheme] = useState<ThemeMode>("white");
  const [documentState, setDocumentState] = useState<StudioDocument>(() =>
    studioDocumentFromConfig(createDefaultModelConfig())
  );
  const [importDraft, setImportDraft] = useState<string>("");
  const [dragOverKey, setDragOverKey] = useState<string | null>(null);
  const [expandedComponentIds, setExpandedComponentIds] = useState<Set<string>>(() => new Set());
  const [expandedMlpStepIds, setExpandedMlpStepIds] = useState<Set<string>>(() => new Set());
  const [expandedRepeatedBlockIds, setExpandedRepeatedBlockIds] = useState<Set<string>>(() => new Set());
  const [lastSavedAt, setLastSavedAt] = useState<number | null>(null);
  const [notice, setNotice] = useState<NoticeState | null>(null);
  const [backendValidation, setBackendValidation] = useState<BackendValidationState>({
    phase: "idle",
    message: "Waiting for edits",
    lastValidatedAt: null,
    warnings: [],
    errors: [],
    normalizedChanged: false,
  });
  const [backendAnalysis, setBackendAnalysis] = useState<BackendAnalysisState>({
    phase: "idle",
    message: "Run backend analysis to instantiate ConfigurableGPT and inspect parameter counts.",
    lastAnalyzedAt: null,
    configSignature: null,
    summary: null,
    warnings: [],
    errors: [],
    instantiationError: null,
  });

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    try {
      const savedTheme = window.localStorage.getItem(THEME_STORAGE_KEY);
      if (savedTheme === "dark" || savedTheme === "white") {
        setTheme(savedTheme);
      }

      const savedImportDraft = window.localStorage.getItem(IMPORT_DRAFT_STORAGE_KEY);
      if (typeof savedImportDraft === "string") {
        setImportDraft(savedImportDraft);
      }

      const raw = window.localStorage.getItem(DOCUMENT_STORAGE_KEY);
      if (!raw) {
        return;
      }

      const parsed = JSON.parse(raw) as unknown;
      const imported = parseImportedModelConfig(parsed);
      if (imported.config) {
        setDocumentState(studioDocumentFromConfig(imported.config));
      }
    } catch {
      // ignore corrupted local storage and continue with defaults
    }
  }, []);

  useEffect(() => {
    if (typeof document !== "undefined") {
      document.documentElement.dataset.theme = theme;
    }
    if (typeof window !== "undefined") {
      window.localStorage.setItem(THEME_STORAGE_KEY, theme);
    }
  }, [theme]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    try {
      const config = studioDocumentToConfig(documentState);
      window.localStorage.setItem(DOCUMENT_STORAGE_KEY, JSON.stringify(config));
      setLastSavedAt(Date.now());
    } catch {
      // no-op
    }
  }, [documentState]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    window.localStorage.setItem(IMPORT_DRAFT_STORAGE_KEY, importDraft);
  }, [importDraft]);

  useEffect(() => {
    if (!notice) {
      return;
    }
    const timeoutId = window.setTimeout(() => {
      setNotice((current) => (current?.at === notice.at ? null : current));
    }, 2800);
    return () => window.clearTimeout(timeoutId);
  }, [notice]);

  useEffect(() => {
    const validComponentIds = new Set(collectAllComponentIds(documentState));
    setExpandedComponentIds((current) => {
      let changed = false;
      const next = new Set<string>();
      current.forEach((id) => {
        if (validComponentIds.has(id)) {
          next.add(id);
        } else {
          changed = true;
        }
      });
      return changed ? next : current;
    });

    const validMlpStepIds = new Set(collectAllMlpStepIds(documentState));
    setExpandedMlpStepIds((current) => {
      let changed = false;
      const next = new Set<string>();
      current.forEach((id) => {
        if (validMlpStepIds.has(id)) {
          next.add(id);
        } else {
          changed = true;
        }
      });
      return changed ? next : current;
    });

    const validBlockIds = new Set(documentState.blocks.map((block) => block.id));
    setExpandedRepeatedBlockIds((current) => {
      let changed = false;
      const next = new Set<string>();
      current.forEach((id) => {
        if (validBlockIds.has(id)) {
          next.add(id);
        } else {
          changed = true;
        }
      });
      return changed ? next : current;
    });
  }, [documentState]);

  const modelConfig = studioDocumentToConfig(documentState);
  const localDiagnostics = validateLocalConfig(modelConfig);
  const localErrors = localDiagnostics.filter((item) => item.level === "error");
  const localWarnings = localDiagnostics.filter((item) => item.level === "warning");
  const metrics = collectBuilderMetrics(documentState);
  const consecutiveBlockGroups = collectConsecutiveIdenticalBlockGroups(documentState.blocks);
  const expandedBlockGroupKeys = new Set(
    consecutiveBlockGroups
      .filter((group) => {
        if (group.count <= 1) {
          return false;
        }
        for (let index = group.startIndex; index <= group.endIndex; index += 1) {
          if (expandedRepeatedBlockIds.has(documentState.blocks[index].id)) {
            return true;
          }
        }
        return false;
      })
      .map((group) => group.key)
  );

  const compactJson = JSON.stringify(modelConfig);
  const deferredJsonSignature = useDeferredValue(compactJson);
  const previewJson = JSON.stringify(JSON.parse(deferredJsonSignature), null, 2);

  useEffect(() => {
    const hasLocalErrors = localErrors.length > 0;
    if (hasLocalErrors) {
      setBackendValidation((current) => ({
        ...current,
        phase: "skipped",
        message: "Backend validation paused until local errors are fixed.",
        warnings: [],
        errors: [],
        normalizedChanged: false,
      }));
      return;
    }

    const runId = validationRunRef.current + 1;
    validationRunRef.current = runId;
    const controller = new AbortController();
    const timeoutId = window.setTimeout(async () => {
      setBackendValidation((current) => ({
        ...current,
        phase: "validating",
        message: "Validating with backend…",
      }));
      try {
        const result = await validateModelConfig(modelConfig, controller.signal);
        if (validationRunRef.current !== runId) {
          return;
        }
        const normalizedChanged =
          JSON.stringify(result.normalized_config) !== JSON.stringify(modelConfig);
        const issueSummary = [
          result.errors.length > 0
            ? `${result.errors.length} backend error${result.errors.length === 1 ? "" : "s"}`
            : null,
          result.warnings.length > 0
            ? `${result.warnings.length} backend warning${result.warnings.length === 1 ? "" : "s"}`
            : null,
        ]
          .filter((part): part is string => part !== null)
          .join(" · ");
        setBackendValidation({
          phase: "success",
          message:
            result.errors.length > 0
              ? `Backend validation found issues${issueSummary ? ` (${issueSummary})` : ""}.`
              : normalizedChanged
                ? `Backend validation passed (normalized config differs)${issueSummary ? ` · ${issueSummary}` : ""}.`
                : issueSummary
                  ? `Backend validation passed · ${issueSummary}.`
                  : "Backend validation passed.",
          lastValidatedAt: Date.now(),
          warnings: result.warnings,
          errors: result.errors,
          normalizedChanged,
        });
      } catch (error) {
        if (controller.signal.aborted || validationRunRef.current !== runId) {
          return;
        }
        setBackendValidation({
          phase: "fallback",
          message:
            error instanceof Error
              ? `Backend validation unavailable: ${error.message}`
              : "Backend validation unavailable; using local checks.",
          lastValidatedAt: Date.now(),
          warnings: [],
          errors: [],
          normalizedChanged: false,
        });
      }
    }, VALIDATION_DEBOUNCE_MS);

    return () => {
      controller.abort();
      window.clearTimeout(timeoutId);
    };
  }, [compactJson, localErrors.length]);

  const backendDiagnostics: Diagnostic[] = [];
  if (backendValidation.phase === "fallback") {
    pushDiagnostic(
      backendDiagnostics,
      "warning",
      "backend",
      "/validate/model",
      backendValidation.message
    );
  }
  if (backendValidation.phase === "success" && backendValidation.normalizedChanged) {
    pushDiagnostic(
      backendDiagnostics,
      "info",
      "backend",
      "/validate/model",
      "Backend returned a normalized config that differs from the current draft."
    );
  }
  backendValidation.errors.forEach((issue) => {
    pushDiagnostic(backendDiagnostics, "error", "backend", issue.path, issue.message);
  });
  backendValidation.warnings.forEach((issue) => {
    pushDiagnostic(backendDiagnostics, "warning", "backend", issue.path, issue.message);
  });

  const diagnostics = [...localDiagnostics, ...backendDiagnostics];
  const totalErrors = diagnostics.filter((item) => item.level === "error").length;
  const totalWarnings = diagnostics.filter((item) => item.level === "warning").length;
  const backendAnalysisStale =
    backendAnalysis.configSignature !== null && backendAnalysis.configSignature !== compactJson;

  const validationStatusLabel =
    totalErrors > 0
      ? `${totalErrors} error${totalErrors === 1 ? "" : "s"}`
      : totalWarnings > 0
        ? `${totalWarnings} warning${totalWarnings === 1 ? "" : "s"}`
        : "Clean";

  function setNoticeMessage(tone: NoticeTone, message: string): void {
    setNotice({ tone, message, at: Date.now() });
  }

  function toggleExpandedComponent(componentId: string): void {
    setExpandedComponentIds((current) => {
      const next = new Set(current);
      if (next.has(componentId)) {
        next.delete(componentId);
      } else {
        next.add(componentId);
      }
      return next;
    });
  }

  function toggleExpandedMlpStep(stepId: string): void {
    setExpandedMlpStepIds((current) => {
      const next = new Set(current);
      if (next.has(stepId)) {
        next.delete(stepId);
      } else {
        next.add(stepId);
      }
      return next;
    });
  }

  function toggleExpandedBlockGroup(groupKey: string): void {
    const group = consecutiveBlockGroups.find((item) => item.key === groupKey);
    if (!group || group.count <= 1) {
      return;
    }
    const groupBlockIds = documentState.blocks
      .slice(group.startIndex, group.endIndex + 1)
      .map((block) => block.id);

    setExpandedRepeatedBlockIds((current) => {
      const next = new Set(current);
      const isExpanded = groupBlockIds.some((id) => next.has(id));
      if (isExpanded) {
        groupBlockIds.forEach((id) => next.delete(id));
      } else {
        groupBlockIds.forEach((id) => next.add(id));
      }
      return next;
    });
  }

  function expandAllCanvasNodes(): void {
    setExpandedRepeatedBlockIds(
      new Set(
        consecutiveBlockGroups.flatMap((group) =>
          group.count > 1
            ? documentState.blocks
                .slice(group.startIndex, group.endIndex + 1)
                .map((block) => block.id)
            : []
        )
      )
    );
    setExpandedComponentIds(new Set(collectAllComponentIds(documentState)));
    setExpandedMlpStepIds(new Set(collectAllMlpStepIds(documentState)));
  }

  function collapseAllCanvasNodes(): void {
    setExpandedRepeatedBlockIds(new Set());
    setExpandedComponentIds(new Set());
    setExpandedMlpStepIds(new Set());
  }

  function writeDragPayload(event: DragEvent, payload: DragPayload): void {
    dragRef.current = payload;
    event.dataTransfer.setData(DND_MIME, JSON.stringify(payload));
    event.dataTransfer.effectAllowed = "move";
  }

  function readDragPayload(event: DragEvent): DragPayload | null {
    if (dragRef.current) {
      return dragRef.current;
    }
    const raw = event.dataTransfer.getData(DND_MIME);
    if (!raw) {
      return null;
    }
    try {
      return JSON.parse(raw) as DragPayload;
    } catch {
      return null;
    }
  }

  function clearDragState(): void {
    dragRef.current = null;
    setDragOverKey(null);
  }

  function beginDragComponent(
    event: DragEvent<HTMLDivElement>,
    fromBlockId: string,
    componentId: string
  ): void {
    event.stopPropagation();
    writeDragPayload(event, { kind: "block-component", fromBlockId, componentId });
  }

  function beginDragMlpStep(
    event: DragEvent<HTMLDivElement>,
    fromBlockId: string,
    fromComponentId: string,
    stepId: string
  ): void {
    event.stopPropagation();
    writeDragPayload(event, { kind: "mlp-step", fromBlockId, fromComponentId, stepId });
  }

  function markDropTarget(event: DragEvent<HTMLDivElement>, key: string): void {
    event.preventDefault();
    setDragOverKey(key);
    event.dataTransfer.dropEffect = "move";
  }

  function insertComponentAt(
    targetBlockId: string,
    insertIndex: number,
    componentKind: StudioComponentKind
  ): void {
    const createdComponent = createDefaultStudioComponent(componentKind);

    setDocumentState((current) => {
      const next = clone(current);
      const targetBlockIndex = findBlockIndex(next, targetBlockId);
      if (targetBlockIndex < 0) {
        return current;
      }
      const targetBlock = next.blocks[targetBlockIndex];
      const targetInsertIndex = clamp(insertIndex, 0, targetBlock.components.length);
      targetBlock.components.splice(targetInsertIndex, 0, createdComponent);
      return next;
    });

    setExpandedComponentIds((current) => new Set([...current, createdComponent.id]));
    if (createdComponent.kind === "mlp") {
      setExpandedMlpStepIds(
        (current) => new Set([...current, ...createdComponent.mlp.sequence.map((step) => step.id)])
      );
    }
  }

  function insertMlpStepAt(
    targetBlockId: string,
    targetComponentId: string,
    insertIndex: number,
    stepKind: MlpStepKind
  ): void {
    const createdStep = createDefaultStudioMlpStep(stepKind);

    setDocumentState((current) => {
      const next = clone(current);
      const targetRef = getMlpComponent(next, targetBlockId, targetComponentId);
      if (!targetRef) {
        return current;
      }
      const targetSequence = targetRef.component.mlp.sequence;
      const targetInsertIndex = clamp(insertIndex, 0, targetSequence.length);
      targetSequence.splice(targetInsertIndex, 0, createdStep);
      return next;
    });

    setExpandedMlpStepIds((current) => new Set([...current, createdStep.id]));
  }

  function handleDropComponent(
    event: DragEvent<HTMLDivElement>,
    targetBlockId: string,
    insertIndex: number
  ): void {
    event.preventDefault();
    const payload = readDragPayload(event);
    clearDragState();
    if (!payload) {
      return;
    }

    if (payload.kind === "palette-component") {
      insertComponentAt(targetBlockId, insertIndex, payload.componentKind);
      return;
    }

    setDocumentState((current) => {
      const next = clone(current);
      const targetBlockIndex = findBlockIndex(next, targetBlockId);
      if (targetBlockIndex < 0) {
        return current;
      }
      const targetBlock = next.blocks[targetBlockIndex];
      const targetInsertIndex = clamp(insertIndex, 0, targetBlock.components.length);

      if (payload.kind !== "block-component") {
        return current;
      }

      const sourceBlockIndex = findBlockIndex(next, payload.fromBlockId);
      if (sourceBlockIndex < 0) {
        return current;
      }
      const sourceBlock = next.blocks[sourceBlockIndex];
      const sourceComponentIndex = findComponentIndex(sourceBlock, payload.componentId);
      if (sourceComponentIndex < 0) {
        return current;
      }
      const [moved] = sourceBlock.components.splice(sourceComponentIndex, 1);
      let adjustedInsertIndex = targetInsertIndex;
      if (
        payload.fromBlockId === targetBlockId &&
        sourceComponentIndex < adjustedInsertIndex
      ) {
        adjustedInsertIndex -= 1;
      }
      adjustedInsertIndex = clamp(adjustedInsertIndex, 0, targetBlock.components.length);
      targetBlock.components.splice(adjustedInsertIndex, 0, moved);
      return next;
    });
  }

  function handleDropMlpStep(
    event: DragEvent<HTMLDivElement>,
    targetBlockId: string,
    targetComponentId: string,
    insertIndex: number
  ): void {
    event.preventDefault();
    const payload = readDragPayload(event);
    clearDragState();
    if (!payload) {
      return;
    }

    if (payload.kind === "palette-mlp-step") {
      insertMlpStepAt(targetBlockId, targetComponentId, insertIndex, payload.stepKind);
      return;
    }

    setDocumentState((current) => {
      const next = clone(current);
      const targetRef = getMlpComponent(next, targetBlockId, targetComponentId);
      if (!targetRef) {
        return current;
      }
      const targetSequence = targetRef.component.mlp.sequence;
      let targetInsertIndex = clamp(insertIndex, 0, targetSequence.length);

      if (payload.kind !== "mlp-step") {
        return current;
      }

      const sourceRef = getMlpComponent(next, payload.fromBlockId, payload.fromComponentId);
      if (!sourceRef) {
        return current;
      }
      const sourceIndex = sourceRef.component.mlp.sequence.findIndex((step) => step.id === payload.stepId);
      if (sourceIndex < 0) {
        return current;
      }

      const [moved] = sourceRef.component.mlp.sequence.splice(sourceIndex, 1);
      if (
        payload.fromBlockId === targetBlockId &&
        payload.fromComponentId === targetComponentId &&
        sourceIndex < targetInsertIndex
      ) {
        targetInsertIndex -= 1;
      }
      targetInsertIndex = clamp(targetInsertIndex, 0, targetRef.component.mlp.sequence.length);
      targetRef.component.mlp.sequence.splice(targetInsertIndex, 0, moved);
      return next;
    });
  }

  function updateBaseField<K extends keyof Pick<StudioDocument, "context_length" | "vocab_size" | "n_embd">>(
    key: K,
    value: number
  ): void {
    setDocumentState((current) => ({ ...current, [key]: value }));
  }

  function insertBlockAt(insertIndex: number, preset: BlockInsertPreset): void {
    setDocumentState((current) => {
      const nextBlocks = current.blocks.slice();
      const nextBlock =
        preset === "empty"
          ? studioBlockFromConfig({ components: [] })
          : studioBlockFromConfig(createDefaultBlockConfig());
      nextBlocks.splice(clamp(insertIndex, 0, nextBlocks.length), 0, nextBlock);
      return { ...current, blocks: nextBlocks };
    });
  }

  function addBlock(): void {
    insertBlockAt(Number.MAX_SAFE_INTEGER, "default");
  }

  function duplicateBlock(blockId: string): void {
    setDocumentState((current) => {
      const index = current.blocks.findIndex((block) => block.id === blockId);
      if (index < 0) {
        return current;
      }
      const duplicate = cloneBlockWithNewIds(current.blocks[index]);
      const nextBlocks = current.blocks.slice();
      nextBlocks.splice(index + 1, 0, duplicate);
      return { ...current, blocks: nextBlocks };
    });
  }

  function deleteBlock(blockId: string): void {
    setDocumentState((current) => {
      if (current.blocks.length <= 1) {
        return current;
      }
      return {
        ...current,
        blocks: current.blocks.filter((block) => block.id !== blockId),
      };
    });
  }

  function resetDefaults(): void {
    setDocumentState(studioDocumentFromConfig(createDefaultModelConfig()));
    setNoticeMessage("info", "Reset to default LLM config template.");
  }

  function removeComponent(blockId: string, componentId: string): void {
    setDocumentState((current) => {
      const next = clone(current);
      const blockIndex = findBlockIndex(next, blockId);
      if (blockIndex < 0) {
        return current;
      }
      next.blocks[blockIndex].components = next.blocks[blockIndex].components.filter(
        (component) => component.id !== componentId
      );
      return next;
    });
  }

  function updateComponent(
    blockId: string,
    componentId: string,
    updater: (component: StudioComponent) => StudioComponent
  ): void {
    setDocumentState((current) => {
      const next = clone(current);
      const blockIndex = findBlockIndex(next, blockId);
      if (blockIndex < 0) {
        return current;
      }
      const componentIndex = findComponentIndex(next.blocks[blockIndex], componentId);
      if (componentIndex < 0) {
        return current;
      }
      next.blocks[blockIndex].components[componentIndex] = updater(
        next.blocks[blockIndex].components[componentIndex]
      );
      return next;
    });
  }

  function updateMlpStep(
    blockId: string,
    componentId: string,
    stepId: string,
    updater: (step: StudioMlpStep) => StudioMlpStep
  ): void {
    setDocumentState((current) => {
      const next = clone(current);
      const mlpRef = getMlpComponent(next, blockId, componentId);
      if (!mlpRef) {
        return current;
      }
      const stepIndex = mlpRef.component.mlp.sequence.findIndex((step) => step.id === stepId);
      if (stepIndex < 0) {
        return current;
      }
      mlpRef.component.mlp.sequence[stepIndex] = updater(mlpRef.component.mlp.sequence[stepIndex]);
      return next;
    });
  }

  function removeMlpStep(blockId: string, componentId: string, stepId: string): void {
    setDocumentState((current) => {
      const next = clone(current);
      const mlpRef = getMlpComponent(next, blockId, componentId);
      if (!mlpRef) {
        return current;
      }
      mlpRef.component.mlp.sequence = mlpRef.component.mlp.sequence.filter((step) => step.id !== stepId);
      return next;
    });
  }

  async function importFromFile(event: ChangeEvent<HTMLInputElement>): Promise<void> {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) {
      return;
    }
    try {
      const text = await file.text();
      setImportDraft(text);
      applyImportText(text);
    } catch {
      setNoticeMessage("error", "Failed to read selected JSON file.");
    }
  }

  function applyImportText(text: string): void {
    try {
      const parsedJson = JSON.parse(text) as unknown;
      const imported = parseImportedModelConfig(parsedJson);
      if (!imported.config) {
        setNoticeMessage(
          "error",
          `Import failed: ${imported.errors.slice(0, 3).join(" ")}`
        );
        return;
      }
      startTransition(() => {
        setDocumentState(studioDocumentFromConfig(imported.config as ModelConfig));
      });
      setNoticeMessage("success", "Imported model config JSON into visual builder.");
    } catch (error) {
      setNoticeMessage(
        "error",
        error instanceof Error ? `Import failed: ${error.message}` : "Import failed."
      );
    }
  }

  function exportJson(): void {
    downloadTextFile("model_config.json", JSON.stringify(modelConfig, null, 2));
    setNoticeMessage("success", "Exported model config JSON.");
  }

  async function copyJson(): Promise<void> {
    try {
      await navigator.clipboard.writeText(JSON.stringify(modelConfig, null, 2));
      setNoticeMessage("success", "Copied JSON to clipboard.");
    } catch {
      setNoticeMessage("error", "Clipboard write failed in this environment.");
    }
  }

  async function runBackendAnalysis(): Promise<void> {
    if (localErrors.length > 0) {
      setNoticeMessage("error", "Resolve local errors before running backend model analysis.");
      return;
    }

    const signature = compactJson;
    setBackendAnalysis((current) => ({
      ...current,
      phase: "running",
      message: "Instantiating ConfigurableGPT on backend…",
      warnings: [],
      errors: [],
      instantiationError: null,
    }));

    try {
      const result = await analyzeModelConfig(modelConfig);
      const analysis = result.analysis;
      const analysisReady = result.instantiated && analysis !== null;
      const hasIssues = result.errors.length > 0;
      setBackendAnalysis({
        phase: analysisReady ? "success" : "error",
        message: analysisReady
          ? `Backend model analysis ready (${analysis.instantiation_time_ms.toFixed(1)} ms).`
          : result.instantiation_error ??
            (hasIssues
              ? "Backend analysis blocked by validation issues."
              : "Backend analysis did not return metrics."),
        lastAnalyzedAt: Date.now(),
        configSignature: signature,
        summary: analysis,
        warnings: result.warnings,
        errors: result.errors,
        instantiationError: result.instantiation_error,
      });
      if (!analysisReady) {
        setNoticeMessage(
          "error",
          result.instantiation_error ?? "Backend model analysis failed."
        );
      }
    } catch (error) {
      setBackendAnalysis({
        phase: "error",
        message:
          error instanceof Error
            ? `Backend analysis unavailable: ${error.message}`
            : "Backend analysis unavailable.",
        lastAnalyzedAt: Date.now(),
        configSignature: signature,
        summary: null,
        warnings: [],
        errors: [],
        instantiationError: error instanceof Error ? error.message : "Unknown error",
      });
    }
  }


  return (
    <StudioPageView
      fileInputRef={fileInputRef}
      importFromFile={importFromFile}
      theme={theme}
      setTheme={setTheme}
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
      documentState={documentState}
      updateBaseField={updateBaseField}
      setDocumentState={setDocumentState}
      clearDragState={clearDragState}
      diagnostics={diagnostics}
      localErrors={localErrors}
      localWarnings={localWarnings}
      backendAnalysis={backendAnalysis}
      backendAnalysisStale={backendAnalysisStale}
      runBackendAnalysis={runBackendAnalysis}
      previewJson={previewJson}
      copyJson={copyJson}
      importDraft={importDraft}
      setImportDraft={setImportDraft}
      applyImportText={applyImportText}
      modelConfig={modelConfig}
      consecutiveBlockGroups={consecutiveBlockGroups}
      dragOverKey={dragOverKey}
      expandedComponentIds={expandedComponentIds}
      expandedMlpStepIds={expandedMlpStepIds}
      expandedBlockGroupKeys={expandedBlockGroupKeys}
      expandAllCanvasNodes={expandAllCanvasNodes}
      collapseAllCanvasNodes={collapseAllCanvasNodes}
      toggleExpandedBlockGroup={toggleExpandedBlockGroup}
      toggleExpandedComponent={toggleExpandedComponent}
      toggleExpandedMlpStep={toggleExpandedMlpStep}
      duplicateBlock={duplicateBlock}
      deleteBlock={deleteBlock}
      removeComponent={removeComponent}
      removeMlpStep={removeMlpStep}
      insertComponentAt={insertComponentAt}
      insertMlpStepAt={insertMlpStepAt}
      updateComponent={updateComponent}
      updateMlpStep={updateMlpStep}
      insertBlockAt={insertBlockAt}
      beginDragComponent={beginDragComponent}
      beginDragMlpStep={beginDragMlpStep}
      markDropTarget={markDropTarget}
      handleDropComponent={handleDropComponent}
      handleDropMlpStep={handleDropMlpStep}
    />
  );
}
