"use client";

import { useThemeMode } from "../../../lib/theme";
import { formatAge, useWorkspaceAssetInventory } from "../../../lib/workspaceAssets";

const AUTO_REFRESH_SECONDS = 30;

export function useWorkspaceHomeController() {
  const [theme, setTheme] = useThemeMode();
  const inventory = useWorkspaceAssetInventory({
    autoRefreshMs: AUTO_REFRESH_SECONDS * 1000,
  });
  const showInitialWorkspaceLoading = inventory.loading && inventory.lastRefreshedAt === null;
  const activeCount =
    inventory.counts.tokenizerRunningCount + inventory.counts.trainingRunningCount;
  const failedCount =
    inventory.counts.tokenizerFailedCount + inventory.counts.trainingFailedCount;

  return {
    theme,
    setTheme,
    inventory,
    showInitialWorkspaceLoading,
    activeCount,
    failedCount,
    syncLabel:
      showInitialWorkspaceLoading || inventory.lastRefreshedAt === null
        ? "Scanning workspace..."
        : `Synced ${formatAge(new Date(inventory.lastRefreshedAt).toISOString())}`,
  };
}
