import { WorkspaceAssetManager } from "../../components/WorkspaceAssetManager";
import styles from "../../workspace-home.module.css";
import { HomeHero } from "./HomeHero";
import { HomeNavigation } from "./HomeNavigation";
import { HomeStatsGrid } from "./HomeStatsGrid";
import { HomeSyncStatus } from "./HomeSyncStatus";
import type { useWorkspaceHomeController } from "../hooks/useWorkspaceHomeController";

type WorkspaceHomeController = ReturnType<typeof useWorkspaceHomeController>;

type WorkspaceHomePageViewProps = {
  controller: WorkspaceHomeController;
};

export function WorkspaceHomePageView({ controller }: WorkspaceHomePageViewProps) {
  const {
    theme,
    setTheme,
    inventory,
    showInitialWorkspaceLoading,
    activeCount,
    failedCount,
    syncLabel,
  } = controller;

  return (
    <main className={styles.homeRoot}>
      <HomeNavigation
        theme={theme}
        onToggleTheme={() => setTheme((previous) => (previous === "dark" ? "white" : "dark"))}
      />

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

      <HomeHero />

      <HomeStatsGrid
        showInitialWorkspaceLoading={showInitialWorkspaceLoading}
        modelCount={inventory.counts.modelCount}
        totalModelBytes={inventory.counts.totalModelBytes}
        tokenizerCompletedCount={inventory.counts.tokenizerCompletedCount}
        trainingCompletedCount={inventory.counts.trainingCompletedCount}
        activeCount={activeCount}
        failedCount={failedCount}
      />

      <WorkspaceAssetManager
        inventory={inventory}
        title="Workspace Assets"
        description="A unified manager for model configs, tokenizers, and model-training runs across your workspace."
      />

      {(showInitialWorkspaceLoading || inventory.lastRefreshedAt) && (
        <HomeSyncStatus
          refreshing={inventory.refreshing}
          showInitialWorkspaceLoading={showInitialWorkspaceLoading}
          syncLabel={syncLabel}
        />
      )}
    </main>
  );
}
