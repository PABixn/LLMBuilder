import {
  FiHardDrive,
  FiLayers,
  FiRefreshCw,
  FiServer,
  FiXCircle,
} from "react-icons/fi";

import { StatusCard } from "../primitives";
import type { BackendAnalysisState } from "../../types";
import { formatBytes, formatCompactCount, formatTimeAgo } from "../../utils/format";

type BackendAnalysisPanelProps = {
  backendAnalysis: BackendAnalysisState;
  backendAnalysisStale: boolean;
  localErrorCount: number;
  runBackendAnalysis: () => Promise<void>;
};

export function BackendAnalysisPanel({
  backendAnalysis,
  backendAnalysisStale,
  localErrorCount,
  runBackendAnalysis,
}: BackendAnalysisPanelProps) {
  const moduleInventoryEntries = backendAnalysis.summary
    ? Object.entries(backendAnalysis.summary.module_counts).sort(
        ([nameA, countA], [nameB, countB]) =>
          countB - countA || nameA.localeCompare(nameB)
      )
    : [];
  const visibleModuleInventoryEntries = moduleInventoryEntries.slice(0, 8);
  const hiddenModuleInventoryCount = Math.max(
    0,
    moduleInventoryEntries.length - visibleModuleInventoryEntries.length
  );
  const backendActivationTotal =
    (backendAnalysis.summary?.activation_component_count ?? 0) +
    (backendAnalysis.summary?.mlp_activation_step_count ?? 0);
  const backendAnalysisPhaseLabel =
    backendAnalysis.phase === "success"
      ? "Ready"
      : backendAnalysis.phase === "error"
        ? "Error"
        : backendAnalysis.phase === "running"
          ? "Running"
          : "Idle";
  const backendAnalysisPhaseTone =
    backendAnalysis.phase === "success"
      ? "good"
      : backendAnalysis.phase === "error"
        ? "error"
        : backendAnalysis.phase === "running"
          ? "warn"
          : "neutral";

  return (
    <section id="model-analysis" className="panelCard analysisPanel">
      <div className="panelHead">
        <div>
          <p className="panelEyebrow">Backend Model Analysis</p>
          <h2>Runtime analysis</h2>
          <p className="panelCopy">
            Instantiates the model on the backend to verify the config and estimate runtime memory.
          </p>
        </div>
        <div className="actionCluster">
          <button
            type="button"
            className="buttonGhost iconOnly"
            onClick={() => {
              void runBackendAnalysis();
            }}
            disabled={backendAnalysis.phase === "running" || localErrorCount > 0}
            aria-label={backendAnalysis.phase === "running" ? "Analysis running" : "Run analysis"}
            title={backendAnalysis.phase === "running" ? "Analysis running" : "Run analysis"}
          >
            <FiServer />
          </button>
        </div>
      </div>

      <div className="analysisMetaRow" aria-label="Runtime analysis status">
        <div className={`analysisMetaItem tone-${backendAnalysisPhaseTone}`}>
          <span className="analysisMetaLabel">Analysis</span>
          <strong className="analysisMetaValue">{backendAnalysisPhaseLabel}</strong>
        </div>
        <div className="analysisMetaItem tone-neutral">
          <span className="analysisMetaLabel">Last Run</span>
          <strong className="analysisMetaValue">
            {formatTimeAgo(backendAnalysis.lastAnalyzedAt)}
          </strong>
        </div>
        {backendAnalysisStale ? (
          <div className="analysisMetaFlag tone-warn">Stale vs Current Draft</div>
        ) : null}
      </div>

      <p className="analysisMessage">{backendAnalysis.message}</p>

      {backendAnalysis.summary ? (
        <>
          <div className="statusGrid analysisStatsGrid">
            <StatusCard
              title="Parameters"
              value={formatCompactCount(backendAnalysis.summary.total_parameters)}
              detail={`${formatBytes(backendAnalysis.summary.parameter_memory_bytes_fp32)} fp32 · ${formatBytes(backendAnalysis.summary.parameter_memory_bytes_bf16)} bf16`}
              tone="good"
              icon={<FiLayers />}
            />
            <StatusCard
              title="KV Cache / Token"
              value={formatBytes(
                backendAnalysis.summary.estimated_kv_cache_bytes_per_token_fp16
              )}
              detail={`${formatBytes(backendAnalysis.summary.estimated_kv_cache_bytes_for_context_fp16)} @ Context Length`}
              tone="neutral"
              icon={<FiHardDrive />}
            />
            <StatusCard
              title="Head Dim"
              value={
                backendAnalysis.summary.min_head_dim === null
                  ? "N/A"
                  : backendAnalysis.summary.min_head_dim ===
                      backendAnalysis.summary.max_head_dim
                    ? `${backendAnalysis.summary.min_head_dim}`
                    : `${backendAnalysis.summary.min_head_dim}-${backendAnalysis.summary.max_head_dim}`
              }
              detail={`${backendAnalysis.summary.attention_component_count} attention components`}
              tone="neutral"
              icon={<FiServer />}
            />
            <StatusCard
              title="Instantiation"
              value={`${backendAnalysis.summary.instantiation_time_ms.toFixed(1)} ms`}
              detail={`${formatCompactCount(backendAnalysis.summary.trainable_parameters)} trainable`}
              tone="neutral"
              icon={<FiRefreshCw />}
            />
          </div>

          <div className="twoColLayout analysisLayout">
            <div className="workflowItem">
              <div className="workflowTitle">Component counts</div>
              <div className="analysisChipRow">
                <span>{backendAnalysis.summary.block_count} Blocks</span>
                <span>{backendAnalysis.summary.component_count} Components</span>
                <span>{backendAnalysis.summary.attention_component_count} Attention</span>
                <span>{backendAnalysis.summary.mlp_component_count} MLP</span>
                <span>{backendAnalysis.summary.norm_component_count} Norm</span>
                <span>{backendActivationTotal} Activations</span>
              </div>
            </div>

            <div className="workflowItem">
              <div className="workflowTitle">Module inventory</div>
              <details className="sectionDisclosure compact">
                <summary className="sectionDisclosureSummary">
                  {moduleInventoryEntries.length} module type
                  {moduleInventoryEntries.length === 1 ? "" : "s"}
                </summary>
                <div className="analysisChipRow analysisModuleChipRow">
                  {visibleModuleInventoryEntries.map(([name, count]) => (
                    <span key={name} className="analysisModuleChip">
                      <code>{name}</code>
                      <strong>{count}</strong>
                    </span>
                  ))}
                  {hiddenModuleInventoryCount > 0 ? (
                    <span className="analysisModuleChip isMeta">
                      +{hiddenModuleInventoryCount} more
                    </span>
                  ) : null}
                </div>
              </details>
            </div>
          </div>
        </>
      ) : null}

      {backendAnalysis.instantiationError ? (
        <div className="diagnosticList">
          <div className="diagnosticItem tone-error" role="listitem">
            <div className="diagnosticIcon">
              <FiXCircle />
            </div>
            <div>
              <div className="diagnosticTitle">Model instantiation failed</div>
              <div className="diagnosticMeta">{backendAnalysis.instantiationError}</div>
            </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}
