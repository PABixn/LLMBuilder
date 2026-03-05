"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import {
  FiActivity,
  FiArchive,
  FiCpu,
  FiFolder,
  FiMoon,
  FiRefreshCw,
  FiSun,
  FiZap,
} from "react-icons/fi";

import { WorkspaceAssetManager } from "./components/WorkspaceAssetManager";
import { formatAge, formatBytes, useWorkspaceAssetInventory } from "../lib/workspaceAssets";
import styles from "./workspace-home.module.css";

type ThemeMode = "white" | "dark";

const THEME_STORAGE_KEY = "llm-studio-theme";
const AUTO_REFRESH_SECONDS = 30;

function readStoredTheme(): ThemeMode {
  if (typeof window === "undefined") {
    return "white";
  }
  try {
    const raw = window.localStorage.getItem(THEME_STORAGE_KEY);
    if (raw === "dark" || raw === "white") {
      return raw;
    }
  } catch {
    // Ignore local storage failures in local workspace mode.
  }
  return "white";
}

export default function WorkspaceHomePage() {
  const [theme, setTheme] = useState<ThemeMode>("white");
  const hasHydratedTheme = useRef(false);
  const inventory = useWorkspaceAssetInventory({
    autoRefreshMs: AUTO_REFRESH_SECONDS * 1000,
  });

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

  const counts = inventory.counts;

  return (
    <main className={styles.homeRoot}>
      <nav className="studioNav">
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
        </div>
      </header>

      <section className={styles.statsGrid}>
        <div className={styles.statCard}>
          <div className={styles.statIcon}>
            <FiFolder />
          </div>
          <div className={styles.statContent}>
            <span className={styles.statLabel}>Models</span>
            <span className={styles.statValue}>{counts.modelCount}</span>
            <span className={styles.statDetail}>{formatBytes(counts.totalModelBytes)}</span>
          </div>
        </div>
        <div className={`${styles.statCard} ${styles.toneGood}`}>
          <div className={styles.statIcon}>
            <FiArchive />
          </div>
          <div className={styles.statContent}>
            <span className={styles.statLabel}>Tokenizers</span>
            <span className={styles.statValue}>{counts.tokenizerCompletedCount}</span>
            <span className={styles.statDetail}>Completed</span>
          </div>
        </div>
        <div className={`${styles.statCard} ${styles.toneWarn}`}>
          <div className={styles.statIcon}>
            <FiActivity />
          </div>
          <div className={styles.statContent}>
            <span className={styles.statLabel}>Active</span>
            <span className={styles.statValue}>{counts.tokenizerRunningCount}</span>
            <span className={styles.statDetail}>Running</span>
          </div>
        </div>
        <div
          className={`${styles.statCard} ${
            counts.tokenizerFailedCount > 0 ? styles.toneError : ""
          }`}
        >
          <div className={styles.statIcon}>
            <FiRefreshCw />
          </div>
          <div className={styles.statContent}>
            <span className={styles.statLabel}>Failed</span>
            <span className={styles.statValue}>{counts.tokenizerFailedCount}</span>
            <span className={styles.statDetail}>Attention required</span>
          </div>
        </div>
      </section>

      <WorkspaceAssetManager
        inventory={inventory}
        title="Workspace Assets"
        description="A unified manager for model configs and tokenizer jobs across your workspace."
      />

      {inventory.lastRefreshedAt ? (
        <div className={styles.syncIndicator}>
          <FiRefreshCw
            className={inventory.refreshing ? styles.refreshIconSpinning : ""}
          />
          <span>
            Synced {formatAge(new Date(inventory.lastRefreshedAt).toISOString())}
          </span>
        </div>
      ) : null}
    </main>
  );
}
