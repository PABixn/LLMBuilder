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
import { HelpTooltip, InfoTooltip } from "../../../shared/components/HelpTooltip";

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
          <h1>
            Design model configs.
            <InfoTooltip label="Model Studio explanation" align="left" width="wide">
              <strong>Model Studio</strong>
              <p>
                Build the architecture JSON used by training. Blocks hold components,
                components define computation, and validation checks whether the final config is usable.
              </p>
            </InfoTooltip>
          </h1>
          <p className="panelCopy">
            Add blocks, edit layers, validate, and export JSON.
          </p>
        </div>
        <div className="heroActions">
          <HelpTooltip label="Add block explanation" content="Adds a transformer block to the visual designer. Blocks are the repeated columns of the model architecture.">
            <button
              type="button"
              className="buttonGhost iconOnly"
              onClick={addBlock}
              aria-label="Add block"
            >
              <FiPlus />
            </button>
          </HelpTooltip>
          <HelpTooltip label="Import file explanation" content="Imports model JSON into the builder. Validation and auto-save run after the imported config is applied.">
            <button
              type="button"
              className="buttonGhost iconOnly"
              onClick={() => fileInputRef.current?.click()}
              aria-label="Import file"
            >
              <FiUpload />
            </button>
          </HelpTooltip>
          <HelpTooltip label="Download model JSON explanation" content="Downloads the current validated model configuration as JSON for reuse outside the app.">
            <button
              type="button"
              className="buttonGhost iconOnly"
              onClick={exportJson}
              aria-label="Download model JSON"
            >
              <FiDownload />
            </button>
          </HelpTooltip>
          <HelpTooltip label="Reset defaults explanation" content="Replaces the current builder state with the default model configuration. Auto-save will then save the new state.">
            <button
              type="button"
              className="buttonGhost iconOnly"
              onClick={resetDefaults}
              aria-label="Reset to defaults"
            >
              <FiRefreshCw />
            </button>
          </HelpTooltip>
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
