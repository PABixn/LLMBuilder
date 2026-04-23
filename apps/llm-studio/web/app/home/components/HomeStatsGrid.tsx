import { FiActivity, FiArchive, FiFolder, FiRefreshCw } from "react-icons/fi";

import { formatBytes } from "../../../lib/workspaceAssets";
import styles from "../../workspace-home.module.css";

type HomeStatsGridProps = {
  showInitialWorkspaceLoading: boolean;
  modelCount: number;
  totalModelBytes: number;
  tokenizerCompletedCount: number;
  trainingCompletedCount: number;
  activeCount: number;
  failedCount: number;
};

export function HomeStatsGrid({
  showInitialWorkspaceLoading,
  modelCount,
  totalModelBytes,
  tokenizerCompletedCount,
  trainingCompletedCount,
  activeCount,
  failedCount,
}: HomeStatsGridProps) {
  return (
    <section className={styles.statsGrid}>
      <div className={styles.statCard}>
        <div className={styles.statIcon}>
          <FiFolder />
        </div>
        <div className={styles.statContent}>
          <span className={styles.statLabel}>Models</span>
          <span className={styles.statValue}>{showInitialWorkspaceLoading ? "..." : modelCount}</span>
          <span className={styles.statDetail}>
            {showInitialWorkspaceLoading ? "Scanning workspace" : formatBytes(totalModelBytes)}
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
            {showInitialWorkspaceLoading ? "..." : tokenizerCompletedCount}
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
            {showInitialWorkspaceLoading ? "..." : trainingCompletedCount}
          </span>
          <span className={styles.statDetail}>
            {showInitialWorkspaceLoading ? "Checking jobs" : "Completed"}
          </span>
        </div>
      </div>

      <div className={`${styles.statCard} ${activeCount > 0 ? styles.toneWarn : ""}`}>
        <div className={styles.statIcon}>
          <FiActivity />
        </div>
        <div className={styles.statContent}>
          <span className={styles.statLabel}>Active</span>
          <span className={styles.statValue}>{showInitialWorkspaceLoading ? "..." : activeCount}</span>
          <span className={styles.statDetail}>
            {showInitialWorkspaceLoading ? "Syncing state" : "Tokenizer + training jobs"}
          </span>
        </div>
      </div>

      <div className={`${styles.statCard} ${failedCount > 0 ? styles.toneError : ""}`}>
        <div className={styles.statIcon}>
          <FiRefreshCw />
        </div>
        <div className={styles.statContent}>
          <span className={styles.statLabel}>Needs Attention</span>
          <span className={styles.statValue}>{showInitialWorkspaceLoading ? "..." : failedCount}</span>
          <span className={styles.statDetail}>
            {showInitialWorkspaceLoading ? "Syncing state" : "Failed or cancelled runs"}
          </span>
        </div>
      </div>
    </section>
  );
}
