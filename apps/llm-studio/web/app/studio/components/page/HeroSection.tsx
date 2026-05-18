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
        : "No validation issues";
  const heroBackendHint =
    backendValidation.phase === "fallback"
      ? "Server unavailable. Local checks only."
      : backendValidation.phase === "validating"
        ? "Checking with server..."
        : null;
  const heroValidationPillTone =
    heroValidationTone === "bad" ? "error" : heroValidationTone;

  return (
    <section className="panelCard heroCard">
      <div className="panelHead heroHead">
        <div>
          <h1>Design model configs.</h1>
          <p className="panelCopy">
            Add blocks, edit layers, validate, and export JSON.
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
            aria-label="Download model JSON"
            title="Download model JSON"
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
          <div className="pillBadge tone-neutral">{metrics.blockCount} blocks</div>
          <div className="pillBadge tone-neutral">{metrics.componentCount} components</div>
          <div className="pillBadge tone-neutral">{metrics.mlpStepCount} MLP steps</div>
        </div>
        <div className="heroMetaLine">
          <span>{heroValidationSummary}</span>
          <span className="heroMetaSeparator" aria-hidden>
            •
          </span>
          <span>Auto-saved {formatTimeAgo(lastSavedAt)}</span>
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
