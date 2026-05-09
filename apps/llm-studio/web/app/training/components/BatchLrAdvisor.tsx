import type { Dispatch, ReactNode, SetStateAction } from "react";
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
};

function TrainingAdvisorInfo({ label, children }: TrainingAdvisorInfoProps) {
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
      <span className="trainingAdvisorTooltip" role="tooltip">
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
};

export function BatchLrAdvisor({
  recommendation,
  selectedRecommendationOptionKey,
  setSelectedRecommendationOptionKey,
  trainingConfig,
  preflightError,
  preflightLoading,
  onApplyRecommendation,
}: BatchLrAdvisorProps) {
  const {
    recommendationConfidenceLabel,
    recommendationConfidenceTone,
    selectedBatchTooltipItems,
    selectedBatchTooltipSummary,
    selectedLearningRateTooltipItems,
    selectedLearningRateTooltipSummary,
    selectedRecommendationIsRecommended,
    selectedRecommendationOption,
  } = buildBatchLrAdvisorViewModel(recommendation, selectedRecommendationOptionKey);

  return (
    <section className="trainingAdvisorCard">
      <div className="trainingAdvisorHead">
        <div>
          <p className="panelEyebrow">Batch And LR Advisor</p>
          <h3>Recommended optimizer step sizing</h3>
          <p className="trainingAdvisorCopy">
            {recommendation
              ? recommendation.summary
              : preflightLoading
                ? "Preflight is recalculating the recommendation from the current model, dataset, runtime, and scheduler settings."
                : "Select a model and tokenizer and let preflight run to see the recommended optimizer-step token batch and learning rate."}
          </p>
        </div>
        {recommendation ? (
          <div className="trainingAdvisorMeta">
            <span className={`pillBadge ${recommendationConfidenceTone}`}>
              {recommendationConfidenceLabel}
            </span>
          </div>
        ) : null}
      </div>

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
          </div>

          <div className="trainingAdvisorFoot">
            <p className="trainingAdvisorNote">
              {selectedRecommendationOption.clear_manual_micro_batch
                ? "Applying this clears the manual micro batch so preflight can auto-size the step."
                : "Applying this keeps the current micro-step behavior compatible with the selected profile."}
            </p>
            <div className="trainingAdvisorMeta">
              <span className="pillBadge tone-neutral">
                {formatInteger(selectedRecommendationOption.estimated_tokens_per_run)} run tokens
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
    </section>
  );
}
