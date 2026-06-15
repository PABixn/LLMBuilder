import Link from "next/link";

import { WorkspaceAssetManager } from "../../components/WorkspaceAssetManager";
import { AppTopNav } from "../../shared/components/AppTopNav";
import { useUiMode } from "../../shared/hooks/useUiMode";
import styles from "../../workspace-home.module.css";
import { HomeHero } from "./HomeHero";
import { HomeSyncStatus } from "./HomeSyncStatus";
import type { useWorkspaceHomeController } from "../hooks/useWorkspaceHomeController";

type WorkspaceHomeController = ReturnType<typeof useWorkspaceHomeController>;

type WorkspaceHomePageViewProps = {
  controller: WorkspaceHomeController;
};

export function WorkspaceHomePageView({ controller }: WorkspaceHomePageViewProps) {
  const [uiMode] = useUiMode();
  const {
    inventory,
    showInitialWorkspaceLoading,
    syncLabel,
  } = controller;

  return (
    <main className={styles.homeRoot}>
      <AppTopNav />

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

      {uiMode === "simple" ? (
        <section className={styles.simpleGuideCard}>
          <div>
            <span>Simple Mode</span>
            <h2>Continue the guided flow</h2>
            <p>
              Models: {inventory.counts.modelCount} · Tokenizers:{" "}
              {inventory.counts.tokenizerCompletedCount} complete · Training runs:{" "}
              {inventory.counts.trainingCompletedCount} complete
            </p>
          </div>
          <Link className={styles.primaryButton} href="/simple">
            Open guide
          </Link>
        </section>
      ) : null}

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
