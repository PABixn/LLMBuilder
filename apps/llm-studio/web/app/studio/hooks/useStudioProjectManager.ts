import { useEffect, useRef, useState, type Dispatch, type SetStateAction } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";

import { createProject, fetchProject, updateProject } from "../../../lib/api";
import type { ModelConfig } from "../../../lib/defaults";
import { upsertCachedWorkspaceProject } from "../../../lib/workspaceAssets";
import { studioDocumentFromConfig } from "../utils/document";
import type { StudioDocument } from "../types";

const AUTO_SAVE_DELAY_MS = 800;

type SetNoticeMessage = (tone: "info" | "success" | "error", message: string) => void;

type UseStudioProjectManagerArgs = {
  modelConfig: ModelConfig;
  setNoticeMessage: SetNoticeMessage;
  replaceDocumentState: (nextDocument: StudioDocument) => void;
};

type DetachProjectOptions = {
  clearName?: boolean;
};

type ProjectSnapshot = {
  projectId: string | null;
  name: string | null;
  config: ModelConfig;
  configSignature: string;
};

type SavedSnapshot = {
  projectId: string | null;
  name: string | null;
  configSignature: string;
};

export interface StudioProjectManager {
  projectName: string;
  setProjectName: Dispatch<SetStateAction<string>>;
  currentProjectId: string | null;
  isProjectLoading: boolean;
  isProjectSaving: boolean;
  createNewProject: () => Promise<void>;
  detachProject: (options?: DetachProjectOptions) => void;
}

function normalizedName(value: string): string | null {
  const trimmed = value.trim();
  return trimmed === "" ? null : trimmed;
}

export function useStudioProjectManager({
  modelConfig,
  setNoticeMessage,
  replaceDocumentState,
}: UseStudioProjectManagerArgs): StudioProjectManager {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const [projectName, setProjectName] = useState("");
  const [currentProjectId, setCurrentProjectId] = useState<string | null>(null);
  const [isProjectSaving, setIsProjectSaving] = useState(false);
  const [isProjectLoading, setIsProjectLoading] = useState(false);
  const saveTimerRef = useRef<number | null>(null);
  const activeControllerRef = useRef<AbortController | null>(null);
  const activeRequestIdRef = useRef(0);
  const lastSavedRef = useRef<SavedSnapshot | null>(null);
  const ignoredUrlProjectIdRef = useRef<string | null>(null);
  const currentProjectIdRef = useRef<string | null>(null);
  const currentNameRef = useRef<string | null>(null);
  const currentConfigSignatureRef = useRef("");
  const configSignature = JSON.stringify(modelConfig);
  const nextName = normalizedName(projectName);
  currentProjectIdRef.current = currentProjectId;
  currentNameRef.current = nextName;
  currentConfigSignatureRef.current = configSignature;

  function clearScheduledSave(): void {
    if (saveTimerRef.current !== null) {
      window.clearTimeout(saveTimerRef.current);
      saveTimerRef.current = null;
    }
  }

  function cancelInFlightSave(): void {
    if (activeControllerRef.current) {
      activeControllerRef.current.abort();
      activeControllerRef.current = null;
    }
  }

  function clearPendingPersistence(): void {
    clearScheduledSave();
    cancelInFlightSave();
  }

  function replaceProjectParam(projectId: string | null): void {
    const params = new URLSearchParams(searchParams.toString());
    if (projectId === null) {
      params.delete("project");
    } else {
      params.set("project", projectId);
    }
    const query = params.toString();
    router.replace(query === "" ? pathname : `${pathname}?${query}`, { scroll: false });
  }

  function createSnapshot(projectId: string | null): ProjectSnapshot {
    return {
      projectId,
      name: nextName,
      config: modelConfig,
      configSignature,
    };
  }

  function markSaved(projectId: string, savedName: string | null, savedConfig: ModelConfig): void {
    lastSavedRef.current = {
      projectId,
      name: savedName,
      configSignature: JSON.stringify(savedConfig),
    };
  }

  function isSavedSnapshot(projectId: string | null, name: string | null): boolean {
    const lastSaved = lastSavedRef.current;
    if (!lastSaved) {
      return false;
    }
    return (
      lastSaved.projectId === projectId &&
      lastSaved.name === name &&
      lastSaved.configSignature === configSignature
    );
  }

  function isSnapshotCurrent(snapshot: ProjectSnapshot): boolean {
    return (
      snapshot.projectId === currentProjectIdRef.current &&
      snapshot.name === currentNameRef.current &&
      snapshot.configSignature === currentConfigSignatureRef.current
    );
  }

  async function persistProject(
    snapshot: ProjectSnapshot,
    mode: "create" | "update"
  ): Promise<void> {
    if (activeControllerRef.current !== null) {
      return;
    }

    const controller = new AbortController();
    const requestId = ++activeRequestIdRef.current;
    activeControllerRef.current = controller;
    setIsProjectSaving(true);

    try {
      const savedProject =
        mode === "create" || snapshot.projectId === null
          ? await createProject(snapshot.name, snapshot.config, controller.signal)
          : await updateProject(
              snapshot.projectId,
              snapshot.name,
              snapshot.config,
              controller.signal
            );

      if (controller.signal.aborted || requestId !== activeRequestIdRef.current) {
        return;
      }

      setCurrentProjectId(savedProject.id);
      setProjectName(savedProject.name ?? "");
      markSaved(savedProject.id, savedProject.name, savedProject.model_config);
      upsertCachedWorkspaceProject(savedProject);
      replaceProjectParam(savedProject.id);
    } catch (error) {
      if (controller.signal.aborted || requestId !== activeRequestIdRef.current) {
        return;
      }

      setNoticeMessage(
        "error",
        error instanceof Error
          ? `Could not save config: ${error.message}`
          : "Could not save config."
      );
    } finally {
      if (activeControllerRef.current === controller) {
        activeControllerRef.current = null;
      }
      if (requestId === activeRequestIdRef.current) {
        setIsProjectSaving(false);
      }
    }
  }

  function detachProject(options: DetachProjectOptions = {}): void {
    clearPendingPersistence();
    lastSavedRef.current = null;
    ignoredUrlProjectIdRef.current = currentProjectIdRef.current;
    setCurrentProjectId(null);
    setIsProjectSaving(false);
    replaceProjectParam(null);
    if (options.clearName) {
      setProjectName("");
    }
  }

  async function createNewProject(): Promise<void> {
    if (isProjectSaving || activeControllerRef.current !== null) {
      return;
    }

    detachProject({ clearName: true });
  }

  useEffect(() => {
    const projectIdFromUrl = searchParams.get("project");
    if (projectIdFromUrl !== ignoredUrlProjectIdRef.current) {
      ignoredUrlProjectIdRef.current = null;
    }
    if (projectIdFromUrl && projectIdFromUrl === ignoredUrlProjectIdRef.current) {
      return;
    }
    if (!projectIdFromUrl || projectIdFromUrl === currentProjectId) {
      return;
    }

    async function loadProject() {
      setIsProjectLoading(true);
      try {
        const project = await fetchProject(projectIdFromUrl as string);
        setCurrentProjectId(project.id);
        setProjectName(project.name ?? "");
        replaceDocumentState(studioDocumentFromConfig(project.model_config));
        markSaved(project.id, project.name, project.model_config);
      } catch (err) {
        setNoticeMessage("error", `Could not load config: ${err instanceof Error ? err.message : "Unknown error"}`);
      } finally {
        setIsProjectLoading(false);
      }
    }

    void loadProject();
  }, [searchParams, currentProjectId, replaceDocumentState, setNoticeMessage]);

  useEffect(() => {
    return () => {
      clearScheduledSave();
    };
  }, []);

  useEffect(() => {
    clearScheduledSave();

    if (isProjectSaving || isProjectLoading) {
      return;
    }

    if (currentProjectId === null && nextName === null) {
      setIsProjectSaving(false);
      return;
    }

    if (isSavedSnapshot(currentProjectId, nextName)) {
      setIsProjectSaving(false);
      return;
    }

    const snapshot = createSnapshot(currentProjectId);

    saveTimerRef.current = window.setTimeout(() => {
      if (!isSnapshotCurrent(snapshot)) {
        return;
      }
      void persistProject(snapshot, snapshot.projectId === null ? "create" : "update");
    }, AUTO_SAVE_DELAY_MS);

    return () => {
      clearScheduledSave();
    };
  }, [configSignature, currentProjectId, isProjectSaving, isProjectLoading, nextName]);

  return {
    projectName,
    setProjectName,
    currentProjectId,
    isProjectLoading,
    isProjectSaving,
    createNewProject,
    detachProject,
  };
}
