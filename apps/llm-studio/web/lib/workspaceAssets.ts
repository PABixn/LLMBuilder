"use client";

import { useEffect, useLayoutEffect, useRef, useState } from "react";

import { deleteProject, fetchProject, fetchProjects, type ProjectSummary, updateProject } from "./api";
import {
  deleteTrainingJob as deleteTokenizerJob,
  fetchTrainingJobs as fetchTokenizerJobs,
  type TrainingJob as TokenizerTrainingJob,
} from "./tokenizerLegacyApi";
import {
  deleteTrainingJob as deleteModelTrainingJob,
  fetchTrainingJobs as fetchModelTrainingJobs,
} from "./training/jobs";
import type { TrainingJob as ModelTrainingJob } from "./training/types";

const DATE_FORMATTER = new Intl.DateTimeFormat("en-US", {
  dateStyle: "medium",
  timeStyle: "short",
});
const WORKSPACE_ASSET_CACHE_KEY = "llm-studio-workspace-asset-cache-v2";
const WORKSPACE_ASSET_CHANGED_EVENT = "llm-studio:workspace-assets-changed";

export type WorkspaceAsset = {
  id: string;
  name: string;
  type: "model" | "tokenizer" | "training_run";
  createdAt: string;
  size?: number;
  downloadPath?: string;
  fileName?: string | null;
  status?: string;
  subtitle?: string;
};

export type WorkspaceAssetCounts = {
  modelCount: number;
  totalModelBytes: number;
  tokenizerCompletedCount: number;
  tokenizerRunningCount: number;
  tokenizerFailedCount: number;
  trainingCompletedCount: number;
  trainingRunningCount: number;
  trainingFailedCount: number;
};

export type WorkspaceAssetInventory = {
  assets: WorkspaceAsset[];
  counts: WorkspaceAssetCounts;
  loading: boolean;
  refreshing: boolean;
  error: string | null;
  lastRefreshedAt: number | null;
  deleteAsset: (asset: WorkspaceAsset) => Promise<void>;
  renameAsset: (asset: WorkspaceAsset, newName: string) => Promise<void>;
  deleteAllAssets: () => Promise<void>;
  refresh: () => Promise<void>;
};

export interface UseWorkspaceAssetInventoryOptions {
  autoRefreshMs?: number;
}

interface WorkspaceAssetCache {
  projects: ProjectSummary[];
  tokenizerJobs: TokenizerTrainingJob[];
  trainingJobs: ModelTrainingJob[];
  lastRefreshedAt: number | null;
}

type WorkspaceAssetChangeDetail =
  | {
      type: "project-upsert";
      project: ProjectSummary;
    }
  | {
      type: "invalidate";
    };

function resolveIssueMessage(label: string, reason: unknown): string {
  if (reason instanceof Error && reason.message.trim() !== "") {
    return `${label}: ${reason.message}`;
  }
  return `${label}: Fetch failed`;
}

function readCachedSnapshot(): WorkspaceAssetCache | null {
  if (typeof window === "undefined") {
    return null;
  }

  try {
    const raw = window.localStorage.getItem(WORKSPACE_ASSET_CACHE_KEY);
    if (!raw) {
      return null;
    }

    const parsed = JSON.parse(raw) as Partial<WorkspaceAssetCache>;
    if (!Array.isArray(parsed.projects) || !Array.isArray(parsed.tokenizerJobs)) {
      return null;
    }

    return {
      projects: (parsed.projects as ProjectSummary[]).filter(p => p && typeof p.id === 'string'),
      tokenizerJobs: (parsed.tokenizerJobs as TokenizerTrainingJob[]).filter(j => j && typeof j.id === 'string'),
      trainingJobs: Array.isArray(parsed.trainingJobs)
        ? (parsed.trainingJobs as ModelTrainingJob[]).filter(j => j && typeof j.id === 'string')
        : [],
      lastRefreshedAt:
        typeof parsed.lastRefreshedAt === "number" ? parsed.lastRefreshedAt : null,
    };
  } catch {
    return null;
  }
}

function writeCachedSnapshot(snapshot: WorkspaceAssetCache): void {
  if (typeof window === "undefined") {
    return;
  }

  try {
    window.localStorage.setItem(WORKSPACE_ASSET_CACHE_KEY, JSON.stringify(snapshot));
  } catch {
    // Ignore cache write failures in local workspace mode.
  }
}

function dispatchWorkspaceAssetChange(detail: WorkspaceAssetChangeDetail): void {
  if (typeof window === "undefined") {
    return;
  }

  window.dispatchEvent(
    new CustomEvent<WorkspaceAssetChangeDetail>(WORKSPACE_ASSET_CHANGED_EVENT, {
      detail,
    })
  );
}

function mergeProjects(
  currentProjects: ProjectSummary[],
  nextProject: ProjectSummary
): ProjectSummary[] {
  const mergedProjects = [
    nextProject,
    ...currentProjects.filter((project) => project.id !== nextProject.id),
  ];

  mergedProjects.sort(
    (left, right) => Date.parse(right.created_at) - Date.parse(left.created_at)
  );

  return mergedProjects;
}

export function upsertCachedWorkspaceProject(project: ProjectSummary): void {
  const cached =
    readCachedSnapshot() ??
    ({
      projects: [],
      tokenizerJobs: [],
      trainingJobs: [],
      lastRefreshedAt: null,
    } satisfies WorkspaceAssetCache);
  const refreshedAt = Date.now();

  writeCachedSnapshot({
    projects: mergeProjects(cached.projects, project),
    tokenizerJobs: cached.tokenizerJobs,
    trainingJobs: cached.trainingJobs,
    lastRefreshedAt: refreshedAt,
  });
  dispatchWorkspaceAssetChange({ type: "project-upsert", project });
}

export function invalidateWorkspaceAssetInventory(): void {
  dispatchWorkspaceAssetChange({ type: "invalidate" });
}

function tokenizerName(job: TokenizerTrainingJob): string {
  const rawName = job.tokenizer_config.name;
  if (typeof rawName === "string" && rawName.trim() !== "") {
    return rawName.trim();
  }

  if (typeof job.artifact_file === "string" && job.artifact_file.trim() !== "") {
    return job.artifact_file.trim();
  }

  return `Tokenizer ${job.id.slice(0, 8)}`;
}

function trainingRunName(job: ModelTrainingJob): string {
  if (typeof job.name === "string" && job.name.trim() !== "") {
    return job.name.trim();
  }
  return `Training ${job.id.slice(0, 8)}`;
}

function buildAssets(
  projects: ProjectSummary[],
  tokenizerJobs: TokenizerTrainingJob[],
  trainingJobs: ModelTrainingJob[]
): WorkspaceAsset[] {
  const modelAssets: WorkspaceAsset[] = projects.map((project) => ({
    id: project.id,
    name: project.name || `Project ${project.id.slice(0, 8)}`,
    type: "model",
    createdAt: project.created_at,
    size: project.size_bytes,
    downloadPath: `/projects/${project.id}/artifact`,
    fileName: project.artifact_file,
    status: "READY",
  }));

  const tokenizerAssets: WorkspaceAsset[] = tokenizerJobs
    .map((job) => ({
      id: job.id,
      name: tokenizerName(job),
      type: "tokenizer",
      createdAt: job.created_at,
      downloadPath: job.status === "completed" ? `/tokenizer/jobs/${job.id}/artifact` : undefined,
      fileName: job.artifact_file,
      status: job.status.toUpperCase(),
      subtitle: "Tokenizer artifact",
    }));

  const trainingAssets: WorkspaceAsset[] = trainingJobs.map((job) => ({
    id: job.id,
    name: trainingRunName(job),
    type: "training_run",
    createdAt: job.created_at,
    size: job.output_size_bytes,
    downloadPath: `/training/jobs/${job.id}/artifact`,
    fileName: job.artifact_bundle_file,
    status: job.status.toUpperCase(),
    subtitle: `${job.project_name} • ${job.tokenizer_name}`,
    }));

  return [...modelAssets, ...tokenizerAssets, ...trainingAssets].sort(
    (left, right) => Date.parse(right.createdAt) - Date.parse(left.createdAt)
  );
}

function buildCounts(
  projects: ProjectSummary[],
  tokenizerJobs: TokenizerTrainingJob[],
  trainingJobs: ModelTrainingJob[]
): WorkspaceAssetCounts {
  let totalModelBytes = 0;
  let tokenizerCompletedCount = 0;
  let tokenizerRunningCount = 0;
  let tokenizerFailedCount = 0;
  let trainingCompletedCount = 0;
  let trainingRunningCount = 0;
  let trainingFailedCount = 0;

  for (const project of projects) {
    totalModelBytes += project.size_bytes;
  }

  for (const job of tokenizerJobs) {
    if (job.status === "completed") {
      tokenizerCompletedCount += 1;
      continue;
    }
    if (job.status === "pending" || job.status === "running") {
      tokenizerRunningCount += 1;
      continue;
    }
    if (job.status === "failed") {
      tokenizerFailedCount += 1;
    }
  }

  for (const job of trainingJobs) {
    if (job.status === "completed") {
      trainingCompletedCount += 1;
      continue;
    }
    if (job.status === "pending" || job.status === "running") {
      trainingRunningCount += 1;
      continue;
    }
    if (job.status === "failed" || job.status === "cancelled") {
      trainingFailedCount += 1;
    }
  }

  return {
    modelCount: projects.length,
    totalModelBytes,
    tokenizerCompletedCount,
    tokenizerRunningCount,
    tokenizerFailedCount,
    trainingCompletedCount,
    trainingRunningCount,
    trainingFailedCount,
  };
}

export function formatDate(isoLike: string): string {
  const timestamp = Date.parse(isoLike);
  if (Number.isNaN(timestamp)) {
    return isoLike;
  }
  return DATE_FORMATTER.format(timestamp);
}

export function formatAge(isoLike: string): string {
  const timestamp = Date.parse(isoLike);
  if (Number.isNaN(timestamp)) {
    return isoLike;
  }

  const diffMs = Date.now() - timestamp;
  const diffSeconds = Math.max(0, Math.floor(diffMs / 1000));

  if (diffSeconds < 60) {
    if (diffSeconds < 5) return "just now";
    return `${diffSeconds}s ago`;
  }

  const diffMinutes = Math.floor(diffSeconds / 60);
  if (diffMinutes < 60) {
    return `${diffMinutes}m ago`;
  }

  const diffHours = Math.floor(diffMinutes / 60);
  if (diffHours < 24) {
    return `${diffHours}h ago`;
  }

  const diffDays = Math.floor(diffHours / 24);
  return `${diffDays}d ago`;
}

export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) {
    return "0 B";
  }
  if (bytes < 1024) {
    return `${bytes} B`;
  }

  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes;
  let unitIndex = -1;

  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }

  return `${value.toFixed(value >= 10 ? 1 : 2)} ${units[unitIndex]}`;
}

export function useWorkspaceAssetInventory(
  options: UseWorkspaceAssetInventoryOptions = {}
): WorkspaceAssetInventory {
  const { autoRefreshMs = 30_000 } = options;
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [tokenizerJobs, setTokenizerJobs] = useState<TokenizerTrainingJob[]>([]);
  const [trainingJobs, setTrainingJobs] = useState<ModelTrainingJob[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastRefreshedAt, setLastRefreshedAt] = useState<number | null>(null);
  const requestIdRef = useRef(0);
  const latestProjectsRef = useRef<ProjectSummary[]>([]);
  const latestTokenizerJobsRef = useRef<TokenizerTrainingJob[]>([]);
  const latestTrainingJobsRef = useRef<ModelTrainingJob[]>([]);

  useEffect(() => {
    latestProjectsRef.current = projects;
  }, [projects]);

  useEffect(() => {
    latestTokenizerJobsRef.current = tokenizerJobs;
  }, [tokenizerJobs]);

  useEffect(() => {
    latestTrainingJobsRef.current = trainingJobs;
  }, [trainingJobs]);

  useLayoutEffect(() => {
    const cached = readCachedSnapshot();
    if (!cached) {
      return;
    }

    setProjects(cached.projects);
    setTokenizerJobs(cached.tokenizerJobs);
    setTrainingJobs(cached.trainingJobs);
    setLastRefreshedAt(cached.lastRefreshedAt);
  }, []);

  async function loadSnapshot(background = false): Promise<void> {
    const requestId = ++requestIdRef.current;

    if (background) {
      setRefreshing(true);
    }

    const [projectsResult, tokenizerJobsResult, trainingJobsResult] = await Promise.allSettled([
      fetchProjects(),
      fetchTokenizerJobs(),
      fetchModelTrainingJobs(),
    ]);

    if (requestId !== requestIdRef.current) {
      return;
    }

    const issues: string[] = [];
    let hasAnySuccess = false;

    if (projectsResult.status === "fulfilled") {
      setProjects(projectsResult.value);
      hasAnySuccess = true;
    } else {
      issues.push(resolveIssueMessage("Models", projectsResult.reason));
    }

    if (tokenizerJobsResult.status === "fulfilled") {
      setTokenizerJobs(tokenizerJobsResult.value);
      hasAnySuccess = true;
    } else {
      issues.push(resolveIssueMessage("Tokenizers", tokenizerJobsResult.reason));
    }

    if (trainingJobsResult.status === "fulfilled") {
      setTrainingJobs(trainingJobsResult.value);
      hasAnySuccess = true;
    } else {
      issues.push(resolveIssueMessage("Training", trainingJobsResult.reason));
    }

    if (hasAnySuccess) {
      const refreshedAt = Date.now();
      const nextProjects =
        projectsResult.status === "fulfilled"
          ? projectsResult.value
          : latestProjectsRef.current;
      const nextTokenizerJobs =
        tokenizerJobsResult.status === "fulfilled"
          ? tokenizerJobsResult.value
          : latestTokenizerJobsRef.current;
      const nextTrainingJobs =
        trainingJobsResult.status === "fulfilled"
          ? trainingJobsResult.value
          : latestTrainingJobsRef.current;
      setLastRefreshedAt(refreshedAt);
      writeCachedSnapshot({
        projects: nextProjects,
        tokenizerJobs: nextTokenizerJobs,
        trainingJobs: nextTrainingJobs,
        lastRefreshedAt: refreshedAt,
      });
    }

    setError(issues.length > 0 ? issues.join(" | ") : null);
    setLoading(false);
    setRefreshing(false);
  }

  useEffect(() => {
    void loadSnapshot(false);

    if (autoRefreshMs <= 0) {
      return () => {
        requestIdRef.current += 1;
      };
    }

    const timer = window.setInterval(() => {
      void loadSnapshot(true);
    }, autoRefreshMs);

    return () => {
      requestIdRef.current += 1;
      window.clearInterval(timer);
    };
  }, [autoRefreshMs]);

  useEffect(() => {
    function handleWorkspaceAssetChange(event: Event): void {
      const detail = (event as CustomEvent<WorkspaceAssetChangeDetail>).detail;
      if (detail.type === "project-upsert") {
        setProjects((current) => mergeProjects(current, detail.project));
        setLastRefreshedAt(Date.now());
        setError(null);
        setLoading(false);
        return;
      }

      requestIdRef.current += 1;
      const cached = readCachedSnapshot();
      if (!cached) {
        return;
      }
      setProjects(cached.projects);
      setTokenizerJobs(cached.tokenizerJobs);
      setTrainingJobs(cached.trainingJobs);
      setLastRefreshedAt(cached.lastRefreshedAt);
      setError(null);
      setLoading(false);
    }

    window.addEventListener(WORKSPACE_ASSET_CHANGED_EVENT, handleWorkspaceAssetChange);
    return () => {
      window.removeEventListener(WORKSPACE_ASSET_CHANGED_EVENT, handleWorkspaceAssetChange);
    };
  }, []);

  async function deleteAsset(asset: WorkspaceAsset) {
    try {
      if (asset.type === "model") {
        await deleteProject(asset.id);
      } else if (asset.type === "tokenizer") {
        await deleteTokenizerJob(asset.id);
      } else {
        await deleteModelTrainingJob(asset.id);
      }
      void loadSnapshot(true);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Delete failed";
      setError(message);
      throw err;
    }
  }

  async function renameAsset(asset: WorkspaceAsset, newName: string) {
    if (asset.type !== "model") {
      throw new Error("Only models can be renamed currently.");
    }
    try {
      // Need to fetch full project detail to get config for updateProject
      const project = await fetchProject(asset.id);
      await updateProject(asset.id, newName, project.model_config);
      void loadSnapshot(true);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Rename failed";
      setError(message);
      throw err;
    }
  }

  async function deleteAllAssets() {
    try {
      setRefreshing(true);
      const allAssets = buildAssets(
        latestProjectsRef.current,
        latestTokenizerJobsRef.current,
        latestTrainingJobsRef.current
      );

      // Delete in parallel
      await Promise.allSettled(
        allAssets.map((asset) =>
          asset.type === "model"
            ? deleteProject(asset.id)
            : asset.type === "tokenizer"
              ? deleteTokenizerJob(asset.id)
              : deleteModelTrainingJob(asset.id)
        )
      );

      void loadSnapshot(true);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Failed to delete all assets";
      setError(message);
      throw err;
    }
  }

  return {
    assets: buildAssets(projects, tokenizerJobs, trainingJobs),
    counts: buildCounts(projects, tokenizerJobs, trainingJobs),
    loading,
    refreshing,
    error,
    lastRefreshedAt,
    deleteAsset,
    renameAsset,
    deleteAllAssets,
    refresh: () => loadSnapshot(true),
  };
}
