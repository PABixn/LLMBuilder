"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  FiActivity,
  FiArchive,
  FiArrowRight,
  FiCpu,
  FiDownload,
  FiFolder,
  FiMoon,
  FiRefreshCw,
  FiSun,
  FiZap,
} from "react-icons/fi";

import { apiBaseUrl, fetchProjects, type ProjectSummary } from "../lib/api";
import {
  artifactDownloadUrl,
  fetchTrainingJobs,
  type TrainingJob,
} from "../lib/tokenizerLegacyApi";
import styles from "./workspace-home.module.css";

type ThemeMode = "white" | "dark";

const THEME_STORAGE_KEY = "llm-studio-theme";
const AUTO_REFRESH_SECONDS = 30;

const DATE_FORMATTER = new Intl.DateTimeFormat("en-US", {
  dateStyle: "medium",
  timeStyle: "short",
});

function readStoredTheme(): ThemeMode {
  if (typeof window === "undefined") return "white";
  try {
    const raw = window.localStorage.getItem(THEME_STORAGE_KEY);
    if (raw === "dark" || raw === "white") return raw;
  } catch {}
  return "white";
}

function formatDate(isoLike: string): string {
  const timestamp = Date.parse(isoLike);
  if (Number.isNaN(timestamp)) return isoLike;
  return DATE_FORMATTER.format(timestamp);
}

function formatAge(isoLike: string): string {
  const timestamp = Date.parse(isoLike);
  if (Number.isNaN(timestamp)) return isoLike;
  const diffMs = Date.now() - timestamp;
  const diffMinutes = Math.max(0, Math.floor(diffMs / 60_000));
  if (diffMinutes < 1) return "just now";
  if (diffMinutes < 60) return `${diffMinutes}m ago`;
  const diffHours = Math.floor(diffMinutes / 60);
  if (diffHours < 24) return `${diffHours}h ago`;
  const diffDays = Math.floor(diffHours / 24);
  return `${diffDays}d ago`;
}

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return "0 B";
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes;
  let unitIndex = -1;
  while (value >= 1024 && unitIndex < units.length - 1) {
    value /= 1024;
    unitIndex += 1;
  }
  return `${value.toFixed(value >= 10 ? 1 : 2)} ${units[unitIndex]}`;
}

function modelArtifactDownloadUrl(projectId: string): string {
  const base = apiBaseUrl();
  const resolvedBase = base === "" ? "/api/v1" : base;
  return `${resolvedBase}/projects/${projectId}/artifact`;
}

function tokenizerName(job: TrainingJob): string {
  const config = (job.tokenizer_config as any) || {};
  if (config.name?.trim()) return config.name.trim();
  if (job.artifact_file?.trim()) return job.artifact_file.trim();
  return `Tokenizer ${job.id.slice(0, 8)}`;
}

export default function WorkspaceHomePage() {
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [theme, setTheme] = useState<ThemeMode>("white");
  const [initialLoading, setInitialLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [lastRefreshedAt, setLastRefreshedAt] = useState<number | null>(null);
  const hasHydratedTheme = useRef(false);

  const loadSnapshot = useCallback(async (background = false) => {
    if (background) setRefreshing(true);
    else setInitialLoading(true);

    const [projectsRes, jobsRes] = await Promise.allSettled([
      fetchProjects(),
      fetchTrainingJobs(),
    ]);

    const issues: string[] = [];
    if (projectsRes.status === "fulfilled") {
      setProjects(projectsRes.value);
    } else {
      issues.push(`Models: ${projectsRes.reason instanceof Error ? projectsRes.reason.message : "Fetch failed"}`);
    }

    if (jobsRes.status === "fulfilled") {
      setJobs(jobsRes.value);
    } else {
      issues.push(`Tokenizers: ${jobsRes.reason instanceof Error ? jobsRes.reason.message : "Fetch failed"}`);
    }

    setLoadError(issues.length > 0 ? issues.join(" | ") : null);
    setLastRefreshedAt(Date.now());
    setRefreshing(false);
    setInitialLoading(false);
  }, []);

  useEffect(() => {
    setTheme(readStoredTheme());
  }, []);

  useEffect(() => {
    if (!hasHydratedTheme.current) {
      hasHydratedTheme.current = true;
      return;
    }
    document.documentElement.dataset.theme = theme;
    localStorage.setItem(THEME_STORAGE_KEY, theme);
  }, [theme]);

  useEffect(() => {
    loadSnapshot();
    const timer = setInterval(() => loadSnapshot(true), AUTO_REFRESH_SECONDS * 1000);
    return () => clearInterval(timer);
  }, [loadSnapshot]);

  const sortedProjects = useMemo(() => 
    [...projects].sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at)), 
  [projects]);

  const completedJobs = useMemo(() => 
    jobs.filter(j => j.status === "completed").sort((a, b) => Date.parse(b.created_at) - Date.parse(a.created_at)),
  [jobs]);

  const activeJobs = useMemo(() => jobs.filter(j => j.status === "pending" || j.status === "running"), [jobs]);
  const failedJobs = useMemo(() => jobs.filter(j => j.status === "failed"), [jobs]);

  const totalBytes = useMemo(() => projects.reduce((acc, p) => acc + p.size_bytes, 0), [projects]);

  return (
    <main className={styles.homeRoot}>
      <nav className="studioNav">
        <div className="studioNavBrand">
          <span className="studioNavDot" />
          <span>LLM Builder</span>
        </div>
        <div className="studioNavLinks">
          <Link className="studioNavLink" href="/" aria-current="page">Home</Link>
          <Link className="studioNavLink" href="/studio">LLM Studio</Link>
          <Link className="studioNavLink" href="/tokenizer">Tokenizer Studio</Link>
        </div>
        <button
          className="themeToggle"
          onClick={() => setTheme(t => t === "dark" ? "white" : "dark")}
        >
          {theme === "dark" ? <FiSun /> : <FiMoon />}
        </button>
      </nav>

      {loadError && (
        <div style={{ background: 'var(--danger-soft)', border: '1px solid var(--danger)', padding: '10px 16px', borderRadius: '12px', color: 'var(--danger)', fontSize: '0.85rem', fontWeight: 600 }}>
          ⚠️ Sync issue: {loadError}
        </div>
      )}

      <header className={styles.heroSection}>
        <h1 className={styles.heroTitle}>Build better models, faster.</h1>
        <p className={styles.heroSubtitle}>
          The all-in-one workspace for designing LLM architectures, training custom tokenizers, 
          and managing your model configurations with ease.
        </p>
        <div className={styles.heroActions}>
          <Link href="/studio" className={styles.primaryButton}>
            <FiZap /> Launch LLM Studio
          </Link>
          <Link href="/tokenizer" className={styles.secondaryButton}>
            <FiCpu /> Tokenizer Studio
          </Link>
        </div>
      </header>

      <section className={styles.statsGrid}>
        <div className={styles.statCard}>
          <div className={styles.statIcon}><FiFolder /></div>
          <div className={styles.statContent}>
            <span className={styles.statLabel}>Model Configs</span>
            <span className={styles.statValue}>{projects.length}</span>
            <span className={styles.statDetail}>{formatBytes(totalBytes)} storage</span>
          </div>
        </div>
        <div className={`${styles.statCard} ${styles.toneGood}`}>
          <div className={styles.statIcon}><FiArchive /></div>
          <div className={styles.statContent}>
            <span className={styles.statLabel}>Artifacts</span>
            <span className={styles.statValue}>{completedJobs.length}</span>
            <span className={styles.statDetail}>Ready to deploy</span>
          </div>
        </div>
        <div className={`${styles.statCard} ${styles.toneWarn}`}>
          <div className={styles.statIcon}><FiActivity /></div>
          <div className={styles.statContent}>
            <span className={styles.statLabel}>Active Jobs</span>
            <span className={styles.statValue}>{activeJobs.length}</span>
            <span className={styles.statDetail}>Running in background</span>
          </div>
        </div>
        <div className={`${styles.statCard} ${failedJobs.length > 0 ? styles.toneError : ""}`}>
          <div className={styles.statIcon}><FiRefreshCw /></div>
          <div className={styles.statContent}>
            <span className={styles.statLabel}>Failed Jobs</span>
            <span className={styles.statValue}>{failedJobs.length}</span>
            <span className={styles.statDetail}>{failedJobs.length > 0 ? "Attention required" : "All systems go"}</span>
          </div>
        </div>
      </section>

      <section>
        <div className={styles.sectionHeader}>
          <div>
            <h2 className={styles.sectionTitle}>Recent Model Configs</h2>
            <p className={styles.sectionSubtitle}>Quickly access and download your latest architecture designs.</p>
          </div>
          <Link href="/studio" className={styles.secondaryButton} style={{ padding: '10px 20px', fontSize: '0.9rem' }}>
            View All <FiArrowRight />
          </Link>
        </div>
        
        <div className={styles.assetGrid}>
          {sortedProjects.slice(0, 3).map(project => (
            <div key={project.id} className={styles.assetCard}>
              <div className={styles.assetHeader}>
                <div className={styles.assetIcon}><FiFolder /></div>
                <a 
                  href={modelArtifactDownloadUrl(project.id)} 
                  download={project.artifact_file}
                  className={styles.downloadButton}
                  title="Download JSON"
                >
                  <FiDownload />
                </a>
              </div>
              <div className={styles.assetMain}>
                <span className={styles.assetName}>{project.name || `Project ${project.id.slice(0, 8)}`}</span>
                <span className={styles.assetMeta}>Created {formatAge(project.created_at)}</span>
              </div>
              <div className={styles.assetFooter}>
                <span className={styles.assetTag}>{formatBytes(project.size_bytes)}</span>
                <span className={styles.assetMeta} style={{ fontSize: '0.7rem' }}>JSON Config</span>
              </div>
            </div>
          ))}
          {sortedProjects.length === 0 && (
            <p className={styles.heroSubtitle}>No model configurations found. Start by creating one in the Studio.</p>
          )}
        </div>
      </section>

      <section>
        <div className={styles.sectionHeader}>
          <div>
            <h2 className={styles.sectionTitle}>Trained Tokenizers</h2>
            <p className={styles.sectionSubtitle}>Manage and download your custom trained tokenizer artifacts.</p>
          </div>
          <Link href="/tokenizer" className={styles.secondaryButton} style={{ padding: '10px 20px', fontSize: '0.9rem' }}>
            View All <FiArrowRight />
          </Link>
        </div>

        <div className={styles.assetGrid}>
          {completedJobs.slice(0, 3).map(job => (
            <div key={job.id} className={styles.assetCard}>
              <div className={styles.assetHeader}>
                <div className={styles.assetIcon}><FiArchive /></div>
                {job.artifact_file && (
                  <a 
                    href={artifactDownloadUrl(job.id)} 
                    download={job.artifact_file}
                    className={styles.downloadButton}
                    title="Download Artifact"
                  >
                    <FiDownload />
                  </a>
                )}
              </div>
              <div className={styles.assetMain}>
                <span className={styles.assetName}>{tokenizerName(job)}</span>
                <span className={styles.assetMeta}>Trained {formatAge(job.created_at)}</span>
              </div>
              <div className={styles.assetFooter}>
                <span className={styles.assetTag}>COMPLETED</span>
                <span className={styles.assetMeta} style={{ fontSize: '0.7rem' }}>{formatDate(job.created_at)}</span>
              </div>
            </div>
          ))}
          {completedJobs.length === 0 && (
            <p className={styles.heroSubtitle}>No tokenizer artifacts found. Train your first tokenizer in Tokenizer Studio.</p>
          )}
        </div>
      </section>

      {lastRefreshedAt && (
        <div className={styles.syncIndicator}>
          <FiRefreshCw className={refreshing ? styles.refreshIconSpinning : ""} />
          <span>Last synced {formatAge(new Date(lastRefreshedAt).toISOString())}</span>
        </div>
      )}
    </main>
  );
}
