"use client";

import Link from "next/link";
import {
  FiActivity,
  FiArchive,
  FiCpu,
  FiFolder,
  FiPlay,
  FiMoon,
  FiRefreshCw,
  FiSun,
  FiZap,
} from "react-icons/fi";

import { WorkspaceAssetManager } from "./components/WorkspaceAssetManager";
import { useThemeMode } from "../lib/theme";
import { formatAge, formatBytes, useWorkspaceAssetInventory } from "../lib/workspaceAssets";
import styles from "./workspace-home.module.css";

const AUTO_REFRESH_SECONDS = 30;

export default function WorkspaceHomePage() {
  const [theme, setTheme] = useThemeMode();
  const inventory = useWorkspaceAssetInventory({
    autoRefreshMs: AUTO_REFRESH_SECONDS * 1000,
  });
  const showInitialWorkspaceLoading = inventory.loading && inventory.lastRefreshedAt === null;

  const counts = inventory.counts;
  const activeCount = counts.tokenizerRunningCount + counts.trainingRunningCount;
  const failedCount = counts.tokenizerFailedCount + counts.trainingFailedCount;

  return (
    <main className={styles.homeRoot}>
      <nav className={`${styles.homeNav} studioNav`}>
        <div className="studioNavBrand">
          <span className="studioNavDot" />
          <span>LLM Builder</span>
        </div>
        <div className="studioNavLinks">
          <Link className="studioNavLink" href="/" aria-current="page">
            Home
          </Link>
          <Link className="studioNavLink" href="/studio">
            LLM Studio
          </Link>
          <Link className="studioNavLink" href="/tokenizer">
            Tokenizer Studio
          </Link>
          <Link className="studioNavLink" href="/training">
            Training
          </Link>
          <Link className="studioNavLink" href="/inference">
            Inference
          </Link>
        </div>
        <button
          className="themeToggle"
          onClick={() => setTheme((previous) => (previous === "dark" ? "white" : "dark"))}
        >
          {theme === "dark" ? <FiSun /> : <FiMoon />}
        </button>
      </nav>

      {inventory.error ? (
        <div
          style={{
            background: "var(--danger-soft)",
            border: "1px solid var(--danger)",
            padding: "10px 16px",
            borderRadius: "12px",
            color: "var(--danger)",
            fontSize: "0.85rem",
            fontWeight: 600,
          }}
        >
          Sync issue: {inventory.error}
        </div>
      ) : null}

      <header className={styles.centeredHeader}>
        <h1 className={styles.heroTitle}>Build better models, faster.</h1>
        <p className={styles.heroSubtitle}>
          The all-in-one workspace for designing LLM architectures, training custom tokenizers,
          and managing your model configurations with ease.
        </p>
        <div className={styles.heroActions}>
          <Link href="/studio" className={styles.primaryButton}>
            <FiZap /> LLM Studio
          </Link>
          <Link href="/tokenizer" className={styles.secondaryButton}>
            <FiCpu /> Tokenizer Studio
          </Link>
          <Link href="/training" className={styles.secondaryButton}>
            <FiActivity /> LLM Training
          </Link>
          <Link href="/inference" className={styles.secondaryButton}>
            <FiPlay /> Inference
          </Link>
        </div>
      </header>

      <section className={styles.statsGrid}>
        <div className={styles.statCard}>
          <div className={styles.statIcon}>
            <FiFolder />
          </div>
          <div className={styles.statContent}>
            <span className={styles.statLabel}>Models</span>
            <span className={styles.statValue}>
              {showInitialWorkspaceLoading ? "..." : counts.modelCount}
            </span>
            <span className={styles.statDetail}>
              {showInitialWorkspaceLoading ? "Scanning workspace" : formatBytes(counts.totalModelBytes)}
            </span>
          </div>
        </div>
        <div className={`${styles.statCard} ${styles.toneGood}`}>
          <div className={styles.statIcon}>
            <FiArchive />
          </div>
          <div className={styles.statContent}>
            <span className={styles.statLabel}>Tokenizers</span>
            <span className={styles.statValue}>
              {showInitialWorkspaceLoading ? "..." : counts.tokenizerCompletedCount}
            </span>
            <span className={styles.statDetail}>
              {showInitialWorkspaceLoading ? "Reading artifacts" : "Completed"}
            </span>
          </div>
        </div>
        <div className={`${styles.statCard} ${styles.toneWarn}`}>
          <div className={styles.statIcon}>
            <FiActivity />
          </div>
          <div className={styles.statContent}>
            <span className={styles.statLabel}>Training Runs</span>
            <span className={styles.statValue}>
              {showInitialWorkspaceLoading ? "..." : counts.trainingCompletedCount}
            </span>
            <span className={styles.statDetail}>
              {showInitialWorkspaceLoading ? "Checking jobs" : "Completed"}
            </span>
          </div>
        </div>
        <div
          className={`${styles.statCard} ${
            activeCount > 0 ? styles.toneWarn : ""
          }`}
        >
          <div className={styles.statIcon}>
            <FiActivity />
          </div>
          <div className={styles.statContent}>
            <span className={styles.statLabel}>Active</span>
            <span className={styles.statValue}>
              {showInitialWorkspaceLoading ? "..." : activeCount}
            </span>
            <span className={styles.statDetail}>
              {showInitialWorkspaceLoading ? "Syncing state" : "Tokenizer + training jobs"}
            </span>
          </div>
        </div>
        <div
          className={`${styles.statCard} ${
            failedCount > 0 ? styles.toneError : ""
          }`}
        >
          <div className={styles.statIcon}>
            <FiRefreshCw />
          </div>
          <div className={styles.statContent}>
            <span className={styles.statLabel}>Needs Attention</span>
            <span className={styles.statValue}>
              {showInitialWorkspaceLoading ? "..." : failedCount}
            </span>
            <span className={styles.statDetail}>
              {showInitialWorkspaceLoading ? "Syncing state" : "Failed or cancelled runs"}
            </span>
          </div>
        </div>
      </section>

      <WorkspaceAssetManager
        inventory={inventory}
        title="Workspace Assets"
        description="A unified manager for model configs, tokenizers, and model-training runs across your workspace."
      />

      {showInitialWorkspaceLoading || inventory.lastRefreshedAt ? (
        <div className={styles.syncIndicator}>
          <FiRefreshCw
            className={
              inventory.refreshing || showInitialWorkspaceLoading ? styles.refreshIconSpinning : ""
            }
          />
          <span>
            {showInitialWorkspaceLoading
              ? "Scanning workspace assets..."
              : `Synced ${formatAge(new Date(inventory.lastRefreshedAt as number).toISOString())}`}
          </span>
        </div>
      ) : null}
    </main>
  );
}
