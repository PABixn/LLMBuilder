import { useEffect, useState, type Dispatch, type SetStateAction } from "react";

import { ACTIVATION_TYPES, createDefaultModelConfig, type BlockComponent } from "../../../lib/defaults";
import type {
  NoticeState,
  NoticeTone,
  StudioComponent,
  StudioComponentKind,
  StudioComponentPrefab,
  StudioDocument,
  ThemeMode,
} from "../types";
import {
  COMPONENT_PREFABS_STORAGE_KEY,
  DOCUMENT_STORAGE_KEY,
  IMPORT_DRAFT_STORAGE_KEY,
  THEME_STORAGE_KEY,
} from "../types";
import {
  collectAllComponentIds,
  collectAllMlpStepIds,
  collectConsecutiveIdenticalBlockGroups,
  createComponentPrefab,
  labelForComponentKind,
  studioComponentToConfig,
  studioDocumentFromConfig,
  studioDocumentToConfig,
} from "../utils/document";
import { parseImportedModelConfig } from "../utils/validation";

type SetIdSet = Dispatch<SetStateAction<Set<string>>>;
const MAX_DOCUMENT_HISTORY_ENTRIES = 100;

type DocumentHistoryState = {
  past: StudioDocument[];
  present: StudioDocument;
  future: StudioDocument[];
};

export interface StudioWorkspaceState {
  theme: ThemeMode;
  setTheme: Dispatch<SetStateAction<ThemeMode>>;
  documentState: StudioDocument;
  setDocumentState: Dispatch<SetStateAction<StudioDocument>>;
  canUndoDocument: boolean;
  canRedoDocument: boolean;
  undoDocument: () => void;
  redoDocument: () => void;
  importDraft: string;
  setImportDraft: Dispatch<SetStateAction<string>>;
  expandedComponentIds: Set<string>;
  setExpandedComponentIds: SetIdSet;
  expandedMlpStepIds: Set<string>;
  setExpandedMlpStepIds: SetIdSet;
  expandedRepeatedBlockIds: Set<string>;
  setExpandedRepeatedBlockIds: SetIdSet;
  lastSavedAt: number | null;
  notice: NoticeState | null;
  setNoticeMessage: (tone: NoticeTone, message: string) => void;
  componentPrefabs: StudioComponentPrefab[];
  saveComponentAsPrefab: (component: StudioComponent) => void;
  updateComponentPrefab: (
    prefabId: string,
    nextName: string,
    nextComponent: StudioComponent,
    options?: { silent?: boolean }
  ) => string | null;
  deleteComponentPrefab: (prefabId: string) => void;
  toggleExpandedComponent: (componentId: string) => void;
  toggleExpandedMlpStep: (stepId: string) => void;
  toggleExpandedBlockGroup: (groupKey: string) => void;
  expandAllCanvasNodes: () => void;
  collapseAllCanvasNodes: () => void;
}

function pruneIdSet(current: Set<string>, validIds: Set<string>): Set<string> {
  let changed = false;
  const next = new Set<string>();

  current.forEach((id) => {
    if (validIds.has(id)) {
      next.add(id);
      return;
    }
    changed = true;
  });

  return changed ? next : current;
}

function isStudioComponentKind(value: unknown): value is StudioComponentKind {
  return value === "attention" || value === "mlp" || value === "norm" || value === "activation";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isValidActivationType(value: unknown): boolean {
  return typeof value === "string" && (ACTIVATION_TYPES as readonly string[]).includes(value);
}

function isValidNormConfig(value: unknown): boolean {
  if (!isRecord(value) || typeof value.type !== "string") {
    return false;
  }
  if (value.type === "layernorm") {
    return true;
  }
  if (value.type !== "rmsnorm") {
    return false;
  }
  return typeof value.learnable_gamma === "boolean";
}

function isValidMlpStepConfig(value: unknown): boolean {
  if (!isRecord(value)) {
    return false;
  }
  if ("linear" in value) {
    return isRecord(value.linear) && typeof value.linear.bias === "boolean";
  }
  if ("norm" in value) {
    return isValidNormConfig(value.norm);
  }
  if ("activation" in value) {
    return isRecord(value.activation) && isValidActivationType(value.activation.type);
  }
  return false;
}

function isValidComponentConfig(kind: StudioComponentKind, value: unknown): value is BlockComponent {
  if (!isRecord(value)) {
    return false;
  }

  if (kind === "attention") {
    if (!("attention" in value) || !isRecord(value.attention)) {
      return false;
    }
    return (
      isFiniteNumber(value.attention.n_head) &&
      isFiniteNumber(value.attention.n_kv_head)
    );
  }

  if (kind === "mlp") {
    if (!("mlp" in value) || !isRecord(value.mlp)) {
      return false;
    }
    if (!isFiniteNumber(value.mlp.multiplier) || !Array.isArray(value.mlp.sequence)) {
      return false;
    }
    return value.mlp.sequence.every(isValidMlpStepConfig);
  }

  if (kind === "norm") {
    return "norm" in value && isValidNormConfig(value.norm);
  }

  return (
    "activation" in value &&
    isRecord(value.activation) &&
    isValidActivationType(value.activation.type)
  );
}

function parseStoredComponentPrefabs(raw: string | null): StudioComponentPrefab[] {
  if (!raw) {
    return [];
  }

  try {
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) {
      return [];
    }

    return parsed.flatMap((item) => {
      if (!item || typeof item !== "object") {
        return [];
      }
      const candidate = item as {
        id?: unknown;
        name?: unknown;
        kind?: unknown;
        component?: unknown;
        createdAt?: unknown;
      };

      if (typeof candidate.id !== "string" || candidate.id.trim() === "") {
        return [];
      }
      if (typeof candidate.name !== "string" || candidate.name.trim() === "") {
        return [];
      }
      if (!isStudioComponentKind(candidate.kind)) {
        return [];
      }
      if (!isValidComponentConfig(candidate.kind, candidate.component)) {
        return [];
      }

      return [
        {
          id: candidate.id,
          name: candidate.name,
          kind: candidate.kind,
          component: candidate.component,
          createdAt: typeof candidate.createdAt === "number" ? candidate.createdAt : Date.now(),
        } satisfies StudioComponentPrefab,
      ];
    });
  } catch {
    return [];
  }
}

function createUniquePrefabName(
  kind: StudioComponentKind,
  currentPrefabs: StudioComponentPrefab[]
): string {
  const base = `${labelForComponentKind(kind)} prefab`;
  const existingNames = new Set(currentPrefabs.map((prefab) => prefab.name.toLowerCase()));
  let index = 1;
  while (existingNames.has(`${base} ${index}`.toLowerCase())) {
    index += 1;
  }
  return `${base} ${index}`;
}

function createUniquePrefabNameFromBase(
  baseName: string,
  currentPrefabs: StudioComponentPrefab[],
  excludePrefabId?: string
): string {
  const normalizedBase = baseName.trim();
  if (!normalizedBase) {
    return "Prefab";
  }

  const existingNames = new Set(
    currentPrefabs
      .filter((prefab) => prefab.id !== excludePrefabId)
      .map((prefab) => prefab.name.toLowerCase())
  );
  if (!existingNames.has(normalizedBase.toLowerCase())) {
    return normalizedBase;
  }

  let suffix = 2;
  while (existingNames.has(`${normalizedBase} (${suffix})`.toLowerCase())) {
    suffix += 1;
  }
  return `${normalizedBase} (${suffix})`;
}

export function useStudioWorkspaceState(): StudioWorkspaceState {
  const [theme, setTheme] = useState<ThemeMode>("white");
  const [documentHistory, setDocumentHistory] = useState<DocumentHistoryState>(() => ({
    past: [],
    present: studioDocumentFromConfig(createDefaultModelConfig()),
    future: [],
  }));
  const [importDraft, setImportDraft] = useState<string>("");
  const [expandedComponentIds, setExpandedComponentIds] = useState<Set<string>>(() => new Set());
  const [expandedMlpStepIds, setExpandedMlpStepIds] = useState<Set<string>>(() => new Set());
  const [expandedRepeatedBlockIds, setExpandedRepeatedBlockIds] = useState<Set<string>>(
    () => new Set()
  );
  const [componentPrefabs, setComponentPrefabs] = useState<StudioComponentPrefab[]>([]);
  const [lastSavedAt, setLastSavedAt] = useState<number | null>(null);
  const [notice, setNotice] = useState<NoticeState | null>(null);
  const documentState = documentHistory.present;
  const canUndoDocument = documentHistory.past.length > 0;
  const canRedoDocument = documentHistory.future.length > 0;

  const setDocumentState: Dispatch<SetStateAction<StudioDocument>> = (nextState) => {
    setDocumentHistory((current) => {
      const nextDocument =
        typeof nextState === "function"
          ? (nextState as (current: StudioDocument) => StudioDocument)(current.present)
          : nextState;

      if (nextDocument === current.present) {
        return current;
      }

      const nextPast = [...current.past, current.present];
      if (nextPast.length > MAX_DOCUMENT_HISTORY_ENTRIES) {
        nextPast.splice(0, nextPast.length - MAX_DOCUMENT_HISTORY_ENTRIES);
      }

      return {
        past: nextPast,
        present: nextDocument,
        future: [],
      };
    });
  };

  function replaceDocumentStateWithoutHistory(nextDocument: StudioDocument): void {
    setDocumentHistory({
      past: [],
      present: nextDocument,
      future: [],
    });
  }

  function undoDocument(): void {
    setDocumentHistory((current) => {
      if (current.past.length === 0) {
        return current;
      }
      const nextPast = current.past.slice(0, -1);
      const previousDocument = current.past[current.past.length - 1];
      return {
        past: nextPast,
        present: previousDocument,
        future: [current.present, ...current.future],
      };
    });
  }

  function redoDocument(): void {
    setDocumentHistory((current) => {
      if (current.future.length === 0) {
        return current;
      }
      const [nextDocument, ...remainingFuture] = current.future;
      const nextPast = [...current.past, current.present];
      if (nextPast.length > MAX_DOCUMENT_HISTORY_ENTRIES) {
        nextPast.splice(0, nextPast.length - MAX_DOCUMENT_HISTORY_ENTRIES);
      }
      return {
        past: nextPast,
        present: nextDocument,
        future: remainingFuture,
      };
    });
  }

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
      if (raw) {
        const parsed = JSON.parse(raw) as unknown;
        const imported = parseImportedModelConfig(parsed);
        if (imported.config) {
          replaceDocumentStateWithoutHistory(studioDocumentFromConfig(imported.config));
        }
      }

      setComponentPrefabs(
        parseStoredComponentPrefabs(window.localStorage.getItem(COMPONENT_PREFABS_STORAGE_KEY))
      );
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
      // ignore storage serialization issues
    }
  }, [documentState]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    window.localStorage.setItem(IMPORT_DRAFT_STORAGE_KEY, importDraft);
  }, [importDraft]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    try {
      window.localStorage.setItem(COMPONENT_PREFABS_STORAGE_KEY, JSON.stringify(componentPrefabs));
    } catch {
      // ignore storage serialization issues
    }
  }, [componentPrefabs]);

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
    setExpandedComponentIds((current) =>
      pruneIdSet(current, new Set(collectAllComponentIds(documentState)))
    );
    setExpandedMlpStepIds((current) =>
      pruneIdSet(current, new Set(collectAllMlpStepIds(documentState)))
    );
    setExpandedRepeatedBlockIds((current) =>
      pruneIdSet(
        current,
        new Set(documentState.blocks.map((block) => block.id))
      )
    );
  }, [documentState]);

  function setNoticeMessage(tone: NoticeTone, message: string): void {
    setNotice({ tone, message, at: Date.now() });
  }

  function saveComponentAsPrefab(component: StudioComponent): void {
    const prefabName = createUniquePrefabName(component.kind, componentPrefabs);
    setComponentPrefabs((current) => [createComponentPrefab(component, prefabName), ...current]);
    setNoticeMessage("success", `Saved ${labelForComponentKind(component.kind)} prefab "${prefabName}".`);
  }

  function updateComponentPrefab(
    prefabId: string,
    nextName: string,
    nextComponent: StudioComponent,
    options?: { silent?: boolean }
  ): string | null {
    const targetPrefab = componentPrefabs.find((item) => item.id === prefabId);
    if (!targetPrefab) {
      setNoticeMessage("error", "The selected prefab could not be found.");
      return null;
    }

    if (targetPrefab.kind !== nextComponent.kind) {
      setNoticeMessage("error", "Prefab kind mismatch. Please keep the same component type.");
      return null;
    }

    const uniqueName = createUniquePrefabNameFromBase(nextName, componentPrefabs, prefabId);
    const nextComponentConfig = studioComponentToConfig(nextComponent);
    const targetSignature = JSON.stringify(targetPrefab.component);
    const nextSignature = JSON.stringify(nextComponentConfig);
    const changed = targetPrefab.name !== uniqueName || targetSignature !== nextSignature;

    if (!changed) {
      return uniqueName;
    }

    setComponentPrefabs((current) =>
      current.map((prefab) =>
        prefab.id !== prefabId
          ? prefab
          : {
              ...prefab,
              name: uniqueName,
              component: nextComponentConfig,
            }
      )
    );
    if (!options?.silent) {
      setNoticeMessage("success", `Updated prefab "${uniqueName}".`);
    }
    return uniqueName;
  }

  function deleteComponentPrefab(prefabId: string): void {
    const prefab = componentPrefabs.find((item) => item.id === prefabId);
    if (!prefab) {
      return;
    }
    setComponentPrefabs((current) => current.filter((item) => item.id !== prefabId));
    setNoticeMessage("info", `Deleted prefab "${prefab.name}".`);
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
    const groups = collectConsecutiveIdenticalBlockGroups(documentState.blocks);
    const group = groups.find((item) => item.key === groupKey);
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
    const groups = collectConsecutiveIdenticalBlockGroups(documentState.blocks);
    setExpandedRepeatedBlockIds(
      new Set(
        groups.flatMap((group) =>
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

  return {
    theme,
    setTheme,
    documentState,
    setDocumentState,
    canUndoDocument,
    canRedoDocument,
    undoDocument,
    redoDocument,
    importDraft,
    setImportDraft,
    expandedComponentIds,
    setExpandedComponentIds,
    expandedMlpStepIds,
    setExpandedMlpStepIds,
    expandedRepeatedBlockIds,
    setExpandedRepeatedBlockIds,
    lastSavedAt,
    notice,
    setNoticeMessage,
    componentPrefabs,
    saveComponentAsPrefab,
    updateComponentPrefab,
    deleteComponentPrefab,
    toggleExpandedComponent,
    toggleExpandedMlpStep,
    toggleExpandedBlockGroup,
    expandAllCanvasNodes,
    collapseAllCanvasNodes,
  };
}
