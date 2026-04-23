import { FiRefreshCw } from "react-icons/fi";

import styles from "../../workspace-home.module.css";

type HomeSyncStatusProps = {
  refreshing: boolean;
  showInitialWorkspaceLoading: boolean;
  syncLabel: string;
};

export function HomeSyncStatus({
  refreshing,
  showInitialWorkspaceLoading,
  syncLabel,
}: HomeSyncStatusProps) {
  return (
    <div className={styles.syncIndicator}>
      <FiRefreshCw
        className={refreshing || showInitialWorkspaceLoading ? styles.refreshIconSpinning : ""}
      />
      <span>{syncLabel}</span>
    </div>
  );
}
