"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { useRouter } from "next/navigation";
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
  const router = useRouter();
  const [theme, setTheme] = useThemeMode();
  const [selectedTrainingModelId, setSelectedTrainingModelId] = useState<string | null>(null);
  const [selectedTrainingTokenizerId, setSelectedTrainingTokenizerId] = useState<string | null>(null);
  const inventory = useWorkspaceAssetInventory({
    autoRefreshMs: AUTO_REFRESH_SECONDS * 1000,
  });
  const showInitialWorkspaceLoading = inventory.loading && inventory.lastRefreshedAt === null;

  const counts = inventory.counts;
  const selectedTrainingModel = useMemo(
    () =>
      inventory.assets.find(
        (asset) => asset.type === "model" && asset.id === selectedTrainingModelId
      ) ?? null,
    [inventory.assets, selectedTrainingModelId]
  );
  const selectedTrainingTokenizer = useMemo(
    () =>
      inventory.assets.find(
        (asset) => asset.type === "tokenizer" && asset.id === selectedTrainingTokenizerId
      ) ?? null,
    [inventory.assets, selectedTrainingTokenizerId]
  );
  const trainingLaunchReady =
    selectedTrainingModel !== null &&
    selectedTrainingTokenizer !== null &&
    selectedTrainingTokenizer.status === "COMPLETED";

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

      <section className={styles.launchpadCard}>
        <div className={styles.launchpadHeader}>
          <div className={styles.sectionLead}>
            <h2 className={styles.sectionTitle}>Training Launchpad</h2>
            <p className={styles.sectionCopy}>
              Pair one saved model config with one completed tokenizer artifact, then open the
              training page with both already selected.
            </p>
          </div>
          <div className={styles.launchpadActions}>
            <button
              type="button"
              className={styles.secondaryButton}
              onClick={() => {
                setSelectedTrainingModelId(null);
                setSelectedTrainingTokenizerId(null);
              }}
              disabled={!selectedTrainingModel && !selectedTrainingTokenizer}
            >
              Clear
            </button>
            <button
              type="button"
              className={styles.primaryButton}
              onClick={() => {
                if (!trainingLaunchReady || !selectedTrainingModel || !selectedTrainingTokenizer) {
                  return;
                }
                router.push(
                  `/training?project=${selectedTrainingModel.id}&tokenizerJob=${selectedTrainingTokenizer.id}`
                );
              }}
              disabled={!trainingLaunchReady}
            >
              <FiPlay /> Open Training Page
            </button>
          </div>
        </div>

        <div className={styles.launchpadGrid}>
          <div className={styles.launchpadSlot}>
            <span className={styles.launchpadLabel}>Selected model</span>
            <strong>{selectedTrainingModel?.name ?? "Choose a model config below"}</strong>
            <span>{selectedTrainingModel?.fileName ?? "Saved model projects appear in Workspace Assets."}</span>
          </div>
          <div className={styles.launchpadSlot}>
            <span className={styles.launchpadLabel}>Selected tokenizer</span>
            <strong>{selectedTrainingTokenizer?.name ?? "Choose a completed tokenizer below"}</strong>
            <span>
              {selectedTrainingTokenizer
                ? `${selectedTrainingTokenizer.status ?? "UNKNOWN"} • ${selectedTrainingTokenizer.fileName ?? "artifact"}`
                : "Only completed tokenizer jobs can launch training."}
            </span>
          </div>
        </div>
      </section>

      <WorkspaceAssetManager
        inventory={inventory}
        title="Workspace Assets"
        description="A unified manager for model configs, tokenizers, and model-training runs across your workspace."
        selectedModelId={selectedTrainingModelId}
        selectedTokenizerId={selectedTrainingTokenizerId}
        onUseAsModel={(asset) => setSelectedTrainingModelId(asset.id)}
        onUseAsTokenizer={(asset) => setSelectedTrainingTokenizerId(asset.id)}
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
