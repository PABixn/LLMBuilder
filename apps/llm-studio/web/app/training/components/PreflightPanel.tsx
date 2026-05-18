import {
  forwardRef,
} from "react";
import {
  FiArchive,
  FiBarChart2,
  FiCpu,
  FiLayers,
} from "react-icons/fi";

import type {
  TrainingFixSuggestion,
  TrainingPreflightResponse,
} from "../../../lib/training/types";
import {
  formatIssueLocation,
  issueTone,
} from "../lib/display";
import { formatInteger } from "../lib/metrics";

interface PreflightPanelProps {
  highlighted: boolean;
  onApplyFix: (fix: TrainingFixSuggestion) => void;
  preflight: TrainingPreflightResponse | null;
  preflightError: string | null;
}

export const PreflightPanel = forwardRef<HTMLElement, PreflightPanelProps>(
  function PreflightPanel(
    {
      highlighted,
      onApplyFix,
      preflight,
      preflightError,
    },
    ref
  ) {
    return (
      <section
        id="settings-preflight"
        ref={ref}
        className={`panelCard trainingPreflightPanel settingsCategoryAnchor ${
          highlighted ? "settingsCategoryAnchor-highlight" : ""
        }`}
      >
        <div className="panelHead">
          <div>
            <h2>Preflight</h2>
            <p className="panelCopy">
              Checks assets, settings, files, and memory before training.
            </p>
          </div>
          {preflight ? (
            <span className={`pillBadge ${preflight.valid ? "tone-good" : "tone-error"}`}>
              {preflight.valid ? "Ready to train" : `${preflight.errors.length} blocking issue${preflight.errors.length === 1 ? "" : "s"}`}
            </span>
          ) : null}
        </div>

        {preflightError ? <div className="trainingIssueCard tone-error">{preflightError}</div> : null}

        {preflight?.compatibility ? (
          <div className="statusGrid">
            <div className={`statusCard ${preflight.valid ? "tone-good" : "tone-bad"}`}>
              <div className="statusCardIcon">
                <FiArchive />
              </div>
              <div>
                <div className="statusCardTitle">Tokenizer vocabulary</div>
                <div className="statusCardValue">{formatInteger(preflight.compatibility.tokenizer_vocab_size)}</div>
                <div className="statusCardDetail">Model vocabulary size: {formatInteger(preflight.compatibility.model_vocab_size)}</div>
              </div>
            </div>
            <div className="statusCard">
              <div className="statusCardIcon">
                <FiLayers />
              </div>
              <div>
                <div className="statusCardTitle">Sequence length</div>
                <div className="statusCardValue">{formatInteger(preflight.compatibility.seq_len)}</div>
                <div className="statusCardDetail">Model context limit: {formatInteger(preflight.compatibility.model_context_length)}</div>
              </div>
            </div>
            <div className="statusCard">
              <div className="statusCardIcon">
                <FiBarChart2 />
              </div>
              <div>
                <div className="statusCardTitle">Micro batch size</div>
                <div className="statusCardValue">
                  {formatInteger(preflight.derived_runtime?.micro_batch_size ?? null)}
                </div>
                <div className="statusCardDetail">
                  Gradient accumulation steps: {formatInteger(preflight.derived_runtime?.grad_accum_steps ?? null)}
                </div>
              </div>
            </div>
            <div className="statusCard">
              <div className="statusCardIcon">
                <FiCpu />
              </div>
              <div>
                <div className="statusCardTitle">Device</div>
                <div className="statusCardValue">{preflight.derived_runtime?.device_type ?? "N/A"}</div>
                <div className="statusCardDetail">
                  Memory-estimated max batch size: {formatInteger((preflight.memory_estimate?.max_batch_size as number | undefined) ?? null)}
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="trainingEmpty">
            Choose a model and tokenizer, then finish settings.
          </div>
        )}

        {preflight?.errors.length ? (
          <div className="trainingIssueList">
            {preflight.errors.map((item) => (
              <div key={`${item.code}-${item.path}`} className={`trainingIssueCard tone-${issueTone(item)}`}>
                <div className="trainingIssueTitle">{item.message}</div>
                <div className="trainingIssueMeta" title={item.path}>{formatIssueLocation(item.path)}</div>
              </div>
            ))}
          </div>
        ) : null}

        {preflight?.warnings.length ? (
          <details className="sectionDisclosure" open>
            <summary className="sectionDisclosureSummary">Warnings</summary>
            <div className="trainingIssueList">
              {preflight.warnings.map((item) => (
                <div key={`${item.code}-${item.path}`} className="trainingIssueCard tone-warning">
                  <div className="trainingIssueTitle">{item.message}</div>
                  <div className="trainingIssueMeta" title={item.path}>{formatIssueLocation(item.path)}</div>
                </div>
              ))}
            </div>
          </details>
        ) : null}

        {preflight?.recommended_fixes.length ? (
          <details className="sectionDisclosure" open>
            <summary className="sectionDisclosureSummary">Suggested fixes</summary>
            <div className="trainingFixList">
              {preflight.recommended_fixes.map((fix) => (
                <div key={fix.code} className="trainingFixCard">
                  <div className="trainingIssueTitle">{fix.label}</div>
                  <div className="trainingIssueMeta">{fix.description}</div>
                  <div className="trainingFixActions">
                    <button type="button" className="buttonGhost buttonSmall" onClick={() => onApplyFix(fix)}>
                      Apply
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </details>
        ) : null}
      </section>
    );
  }
);
