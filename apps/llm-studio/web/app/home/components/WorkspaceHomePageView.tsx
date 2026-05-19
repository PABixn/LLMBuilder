import { WorkspaceAssetManager } from "../../components/WorkspaceAssetManager";
import styles from "../../workspace-home.module.css";
import { HomeHero } from "./HomeHero";
import { HomeNavigation } from "./HomeNavigation";
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
          Sync problem: {inventory.error}
        </div>
      ) : null}

      <HomeHero />

      <WorkspaceAssetManager
        inventory={inventory}
        title="Workspace"
        description="Your models, tokenizers, and training runs."
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
