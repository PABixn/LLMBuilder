import type { RefObject } from "react";
import {
  FiDownload,
  FiPlus,
  FiRefreshCw,
  FiUpload,
} from "react-icons/fi";

import type {
  BackendValidationState,
  BuilderMetrics,
  NoticeState,
} from "../../types";
import { formatTimeAgo } from "../../utils/format";

type HeroSectionProps = {
  fileInputRef: RefObject<HTMLInputElement | null>;
  addBlock: () => void;
  exportJson: () => void;
  resetDefaults: () => void;
  notice: NoticeState | null;
  validationStatusLabel: string;
  backendValidation: BackendValidationState;
  totalErrors: number;
  totalWarnings: number;
  metrics: BuilderMetrics;
  lastSavedAt: number | null;
};

export function HeroSection({
  fileInputRef,
  addBlock,
  exportJson,
  resetDefaults,
  notice,
  validationStatusLabel,
  backendValidation,
  totalErrors,
  totalWarnings,
  metrics,
  lastSavedAt,
}: HeroSectionProps) {
  const heroValidationTone =
    totalErrors > 0
      ? "bad"
      : totalWarnings > 0 || backendValidation.phase === "fallback"
        ? "warn"
        : "good";
  const heroValidationSummary =
    totalErrors > 0
      ? `${totalErrors} error${totalErrors === 1 ? "" : "s"} · ${totalWarnings} warning${totalWarnings === 1 ? "" : "s"}`
      : totalWarnings > 0
        ? `${totalWarnings} warning${totalWarnings === 1 ? "" : "s"}`
        : "No validation warnings";
  const heroBackendHint =
    backendValidation.phase === "fallback"
      ? "Backend unavailable (local checks only)"
      : backendValidation.phase === "validating"
        ? "Backend validation running"
        : null;
  const heroValidationPillTone =
    heroValidationTone === "bad" ? "error" : heroValidationTone;

  return (
    <section className="panelCard heroCard">
      <div className="panelHead heroHead">
        <div>
          <h1>Design transformer configs visually.</h1>
          <p className="panelCopy">
            Build blocks, edit components, reorder, validate, and export JSON from one place.
          </p>
        </div>
        <div className="heroActions">
          <button
            type="button"
            className="buttonGhost iconOnly"
            onClick={addBlock}
            aria-label="Add block"
            title="Add block"
          >
            <FiPlus />
          </button>
          <button
            type="button"
            className="buttonGhost iconOnly"
            onClick={() => fileInputRef.current?.click()}
            aria-label="Import file"
            title="Import file"
          >
            <FiUpload />
          </button>
          <button
            type="button"
            className="buttonGhost iconOnly"
            onClick={exportJson}
            aria-label="Save model JSON (download)"
            title="Save model JSON (download)"
          >
            <FiDownload />
          </button>
          <button
            type="button"
            className="buttonGhost iconOnly"
            onClick={resetDefaults}
            aria-label="Reset to defaults"
            title="Reset to defaults"
          >
            <FiRefreshCw />
          </button>
        </div>
      </div>

      {notice ? (
        <div className={`inlineNotice tone-${notice.tone}`} role="status" aria-live="polite">
          {notice.message}
        </div>
      ) : null}

      <div className="heroMetaRow" aria-label="Workspace summary">
        <div className="heroMetaPills">
          <div className={`pillBadge tone-${heroValidationPillTone}`}>
            Validation: {validationStatusLabel}
          </div>
          <div className="pillBadge tone-neutral">{metrics.blockCount} Blocks</div>
          <div className="pillBadge tone-neutral">{metrics.componentCount} Components</div>
          <div className="pillBadge tone-neutral">{metrics.mlpStepCount} MLP Steps</div>
        </div>
        <div className="heroMetaLine">
          <span>{heroValidationSummary}</span>
          <span className="heroMetaSeparator" aria-hidden>
            •
          </span>
          <span>Auto-saved {formatTimeAgo(lastSavedAt)}</span>
          <span className="heroMetaSeparator" aria-hidden>
            •
          </span>
          <span>Saved in this browser; use Save/Download to keep a file.</span>
          {heroBackendHint ? (
            <>
              <span className="heroMetaSeparator" aria-hidden>
                •
              </span>
              <span>{heroBackendHint}</span>
            </>
          ) : null}
        </div>
      </div>
    </section>
  );
}
