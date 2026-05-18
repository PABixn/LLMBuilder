import type { Dispatch, ReactNode, SetStateAction } from "react";
import { FiRefreshCw } from "react-icons/fi";
import type {
  TrainingBatchLrRecommendation,
  TrainingBatchLrRecommendationOption,
} from "../../../lib/training/types";
import { numbersRoughlyEqual } from "../lib/display";
import { buildBatchLrAdvisorViewModel } from "../lib/batchLrAdvisor";
import { formatInteger } from "../lib/metrics";
import { asNumber, asRecord } from "../lib/object";
import { compactWorkflowMessage, formatLearningRate } from "../lib/run";

type TrainingAdvisorInfoProps = {
  label: string;
  children: ReactNode;
  tooltipClassName?: string;
};

function TrainingAdvisorInfo({
  label,
  children,
  tooltipClassName = "",
}: TrainingAdvisorInfoProps) {
  return (
    <span className="trainingAdvisorInfo">
      <button
        type="button"
        className="trainingAdvisorInfoTrigger"
        aria-label={label}
        title={label}
      >
        <span aria-hidden="true">i</span>
      </button>
      <span className={`trainingAdvisorTooltip ${tooltipClassName}`.trim()} role="tooltip">
        {children}
      </span>
    </span>
  );
}

type BatchLrAdvisorProps = {
  recommendation: TrainingBatchLrRecommendation | null;
  selectedRecommendationOptionKey: string | null;
  setSelectedRecommendationOptionKey: Dispatch<SetStateAction<string | null>>;
  trainingConfig: Record<string, unknown>;
  preflightError: string | null;
  preflightLoading: boolean;
  onApplyRecommendation: (option: TrainingBatchLrRecommendationOption) => void;
  onRefreshRecommendation: () => void;
};

export function BatchLrAdvisor({
  recommendation,
  selectedRecommendationOptionKey,
  setSelectedRecommendationOptionKey,
  trainingConfig,
  preflightError,
  preflightLoading,
  onApplyRecommendation,
  onRefreshRecommendation,
}: BatchLrAdvisorProps) {
  const {
    highlightedRecommendationFactors,
    recommendationConfidenceLabel,
    recommendationConfidenceTone,
    selectedBatchTooltipItems,
    selectedBatchTooltipSummary,
    selectedLearningRateTooltipItems,
    selectedLearningRateTooltipSummary,
    selectedRecommendationIsRecommended,
    selectedRecommendationOption,
    selectedStepTooltipItems,
    selectedStepTooltipSummary,
  } = buildBatchLrAdvisorViewModel(recommendation, selectedRecommendationOptionKey);

  return (
    <details className="trainingAdvisorCard" open>
      <summary className="trainingAdvisorSummary">
        <span>
          <span className="panelEyebrow">Advisor</span>
          <span className="trainingAdvisorSummaryTitle">Recommended batch and learning rate</span>
        </span>
        <span className="trainingAdvisorSummaryActions">
          {recommendation ? (
            <span className={`pillBadge ${recommendationConfidenceTone}`}>
              {recommendationConfidenceLabel}
            </span>
          ) : null}
          <button
            type="button"
            className="buttonGhost iconOnly"
            onClick={(event) => {
              event.preventDefault();
              event.stopPropagation();
              onRefreshRecommendation();
            }}
            disabled={preflightLoading}
            aria-label={preflightLoading ? "Refreshing recommendation" : "Refresh recommendation"}
            title={preflightLoading ? "Refreshing recommendation" : "Refresh recommendation"}
          >
            <FiRefreshCw aria-hidden="true" />
          </button>
        </span>
      </summary>
      {!recommendation ? (
        <div className="trainingAdvisorHead">
          <div>
            <p className="trainingAdvisorCopy">
              {preflightLoading
                ? "Updating the recommendation..."
                : "Run preflight to get a batch size and learning rate."}
            </p>
          </div>
        </div>
      ) : null}

      {recommendation && selectedRecommendationOption ? (
        <>
          <div className="trainingAdvisorToolbar">
            <div className="trainingAdvisorProfileRow">
              <span className="trainingAdvisorToolbarLabel">Profile</span>
              <div
                className="modeSwitch trainingAdvisorModeSwitch"
                role="list"
                aria-label="Batch and LR recommendation profiles"
              >
                {recommendation.options.map((option) => {
                  const isSelected = option.key === selectedRecommendationOption.key;
                  return (
                    <button
                      key={option.key}
                      type="button"
                      className={`modeSwitchButton ${
                        isSelected ? "modeSwitchButton-active" : ""
                      }`}
                      onClick={() => setSelectedRecommendationOptionKey(option.key)}
                      aria-pressed={isSelected}
                      title={option.description}
                    >
                      {option.label}
                    </button>
                  );
                })}
              </div>
              <span
                className={`pillBadge ${
                  selectedRecommendationIsRecommended ? "tone-good" : "tone-neutral"
                }`}
              >
                {selectedRecommendationIsRecommended ? "Recommended" : "Alternate"}
              </span>
            </div>
            <button
              type="button"
              className="buttonPrimary"
              onClick={() => onApplyRecommendation(selectedRecommendationOption)}
            >
              Apply recommendation
            </button>
          </div>

          <div className="trainingAdvisorCompactGrid">
            <article className="trainingAdvisorKeyStat">
              <div className="trainingAdvisorStatLabel">
                <span>Full batch size</span>
                <TrainingAdvisorInfo label="Batch sizing details">
                  <strong>{selectedRecommendationOption.label}</strong>
                  <p>{selectedBatchTooltipSummary}</p>
                  <div className="trainingAdvisorTooltipList">
                    {selectedBatchTooltipItems.map((item) => (
                      <div key={item.label} className="trainingAdvisorTooltipItem">
                        <span>{item.label}</span>
                        <strong>{item.detail}</strong>
                      </div>
                    ))}
                  </div>
                </TrainingAdvisorInfo>
              </div>
              <strong>{formatInteger(selectedRecommendationOption.total_batch_size)} tokens</strong>
              <small>
                Current {formatInteger(asNumber(trainingConfig.total_batch_size, 0))} tokens
                {numbersRoughlyEqual(
                  selectedRecommendationOption.total_batch_size,
                  asNumber(trainingConfig.total_batch_size, 0)
                )
                  ? " • already set"
                  : ""}
              </small>
            </article>

            <article className="trainingAdvisorKeyStat">
              <div className="trainingAdvisorStatLabel">
                <span>Base learning rate</span>
                <TrainingAdvisorInfo label="Learning rate details">
                  <strong>{selectedRecommendationOption.label}</strong>
                  <p>{selectedLearningRateTooltipSummary}</p>
                  <div className="trainingAdvisorTooltipList">
                    {selectedLearningRateTooltipItems.map((item) => (
                      <div key={item.label} className="trainingAdvisorTooltipItem">
                        <span>{item.label}</span>
                        <strong>{item.detail}</strong>
                      </div>
                    ))}
                  </div>
                </TrainingAdvisorInfo>
              </div>
              <strong>{formatLearningRate(selectedRecommendationOption.learning_rate)}</strong>
              <small>
                Current {formatLearningRate(asNumber(asRecord(trainingConfig.optimizer).lr, 0.0003))}
                {numbersRoughlyEqual(
                  selectedRecommendationOption.learning_rate,
                  asNumber(asRecord(trainingConfig.optimizer).lr, 0.0003),
                  1e-9
                )
                  ? " • already set"
                  : ""}
              </small>
            </article>

            <article className="trainingAdvisorKeyStat">
              <div className="trainingAdvisorStatLabel">
                <span>Maximum training steps</span>
                <TrainingAdvisorInfo label="Training-step recommendation details">
                  <strong>{selectedRecommendationOption.label}</strong>
                  <p>{selectedStepTooltipSummary}</p>
                  <div className="trainingAdvisorTooltipList">
                    {selectedStepTooltipItems.map((item) => (
                      <div key={item.label} className="trainingAdvisorTooltipItem">
                        <span>{item.label}</span>
                        <strong>{item.detail}</strong>
                      </div>
                    ))}
                  </div>
                </TrainingAdvisorInfo>
              </div>
              <strong>{formatInteger(selectedRecommendationOption.recommended_max_steps)}</strong>
              <small>
                Current {formatInteger(asNumber(trainingConfig.max_steps, 0))}
                {numbersRoughlyEqual(
                  selectedRecommendationOption.recommended_max_steps,
                  asNumber(trainingConfig.max_steps, 0)
                )
                  ? " • already set"
                  : ""}
              </small>
            </article>
          </div>

          {highlightedRecommendationFactors.length > 0 ? (
            <div className="trainingAdvisorReasonControl">
              <span className="trainingAdvisorToolbarLabel">Why this size</span>
              <TrainingAdvisorInfo
                label="Why this optimizer-step size"
                tooltipClassName="trainingAdvisorTooltip-wide"
              >
                <strong>Optimizer-step rationale</strong>
                <p>
                  The advisor reconciles model scale with the active data and memory constraints.
                </p>
                <div className="trainingAdvisorTooltipList">
                  {highlightedRecommendationFactors.map((factor) => (
                    <div
                      key={factor.code}
                      className={`trainingAdvisorTooltipItem tone-${factor.tone}`}
                    >
                      <span>{factor.label}</span>
                      <strong>{factor.detail}</strong>
                    </div>
                  ))}
                </div>
              </TrainingAdvisorInfo>
            </div>
          ) : null}

          <div className="trainingAdvisorFoot">
            <p className="trainingAdvisorNote">
              {selectedRecommendationOption.clear_manual_micro_batch
                ? "Applying this updates total batch size, learning rate, maximum training steps, refits the scheduler, and clears the manual micro batch so preflight can auto-size the step."
                : "Applying this updates total batch size, learning rate, and maximum training steps while refitting the scheduler for the selected profile."}
            </p>
            <div className="trainingAdvisorMeta">
              <span className="pillBadge tone-neutral">
                {formatInteger(
                  selectedRecommendationOption.estimated_tokens_per_recommended_run
                )}{" "}
                recommended run tokens
              </span>
            </div>
          </div>
        </>
      ) : (
        <div className="trainingAdvisorEmpty">
          {preflightError
            ? compactWorkflowMessage(preflightError)
            : "The advisor appears here once preflight can evaluate the current runtime."}
        </div>
      )}
    </details>
  );
}
