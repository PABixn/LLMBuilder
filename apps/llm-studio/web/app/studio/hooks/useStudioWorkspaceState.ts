import { useEffect, useState, type Dispatch, type SetStateAction } from "react";

import { createDefaultModelConfig } from "../../../lib/defaults";
import type {
  NoticeState,
  NoticeTone,
  StudioDocument,
  ThemeMode,
} from "../types";
import {
  DOCUMENT_STORAGE_KEY,
  IMPORT_DRAFT_STORAGE_KEY,
  THEME_STORAGE_KEY,
} from "../types";
import {
  collectAllComponentIds,
  collectAllMlpStepIds,
  collectConsecutiveIdenticalBlockGroups,
  studioDocumentFromConfig,
  studioDocumentToConfig,
} from "../utils/document";
import { parseImportedModelConfig } from "../utils/validation";

type SetIdSet = Dispatch<SetStateAction<Set<string>>>;

export interface StudioWorkspaceState {
  theme: ThemeMode;
  setTheme: Dispatch<SetStateAction<ThemeMode>>;
  documentState: StudioDocument;
  setDocumentState: Dispatch<SetStateAction<StudioDocument>>;
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

export function useStudioWorkspaceState(): StudioWorkspaceState {
  const [theme, setTheme] = useState<ThemeMode>("white");
  const [documentState, setDocumentState] = useState<StudioDocument>(() =>
    studioDocumentFromConfig(createDefaultModelConfig())
  );
  const [importDraft, setImportDraft] = useState<string>("");
  const [expandedComponentIds, setExpandedComponentIds] = useState<Set<string>>(() => new Set());
  const [expandedMlpStepIds, setExpandedMlpStepIds] = useState<Set<string>>(() => new Set());
  const [expandedRepeatedBlockIds, setExpandedRepeatedBlockIds] = useState<Set<string>>(
    () => new Set()
  );
  const [lastSavedAt, setLastSavedAt] = useState<number | null>(null);
  const [notice, setNotice] = useState<NoticeState | null>(null);

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
    toggleExpandedComponent,
    toggleExpandedMlpStep,
    toggleExpandedBlockGroup,
    expandAllCanvasNodes,
    collapseAllCanvasNodes,
  };
}
