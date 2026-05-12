import {
  forwardRef,
  type Dispatch,
  type Ref,
  type SetStateAction,
} from "react";
import type {
  TrainingBatchLrRecommendation,
  TrainingBatchLrRecommendationOption,
} from "../../../lib/training/types";
import {
  ConfigNumberInput,
  OptionalConfigNumberInput,
} from "../../shared/components/ConfigNumberInput";
import { asNumber, asRecord } from "../lib/object";
import { BatchLrAdvisor } from "./BatchLrAdvisor";
import { LearningRateSchedulePlanner } from "./LearningRateSchedulePlanner";

type TrainingPlanPanelProps = {
  dataloaderConfig: Record<string, unknown>;
  handleDataloaderField: (path: string[], value: unknown) => void;
  handleLrSchedulersChange: (schedulers: Record<string, unknown>[]) => void;
  handleMaxStepsChange: (value: number) => void;
  handleOptionalTrainingField: (path: string[], value: unknown | null) => void;
  handleTrainingField: (path: string[], value: unknown) => void;
  highlighted: boolean;
  onApplyRecommendation: (option: TrainingBatchLrRecommendationOption) => void;
  onRefreshRecommendation: () => void;
  preflightError: string | null;
  preflightLoading: boolean;
  recommendation: TrainingBatchLrRecommendation | null;
  selectedRecommendationOptionKey: string | null;
  setSelectedRecommendationOptionKey: Dispatch<SetStateAction<string | null>>;
  trainingConfig: Record<string, unknown>;
  trainingSettingsRef: Ref<HTMLDivElement>;
};

export const TrainingPlanPanel = forwardRef<HTMLDetailsElement, TrainingPlanPanelProps>(
  function TrainingPlanPanel(
    {
      dataloaderConfig,
      handleDataloaderField,
      handleLrSchedulersChange,
      handleMaxStepsChange,
      handleOptionalTrainingField,
      handleTrainingField,
      highlighted,
      onApplyRecommendation,
      onRefreshRecommendation,
      preflightError,
      preflightLoading,
      recommendation,
      selectedRecommendationOptionKey,
      setSelectedRecommendationOptionKey,
      trainingConfig,
      trainingSettingsRef,
    },
    ref
  ) {
    return (
      <details className="settingsPanel trainingPlanSettingsPanel" open ref={ref}>
        <summary>Training plan</summary>
        <div className="settingsGrid">
          <div
            id="settings-training"
            ref={trainingSettingsRef}
            className={`settingsGroup settingsCategoryAnchor ${
              highlighted ? "settingsCategoryAnchor-highlight" : ""
            }`}
          >
            <div className="settingsGroupHeader">
              <h3>Core launch knobs</h3>
              <p className="settingsGroupHint">
                Tune the values you are most likely to touch between runs before opening the
                deeper runtime controls.
              </p>
            </div>
            <div className="fieldGrid trainingSettingsCompactGrid">
              <label className="fieldLabel">
                <span>Sequence length</span>
                <ConfigNumberInput
                  value={asNumber(trainingConfig.seq_len, 128)}
                  onCommit={(value) => handleTrainingField(["seq_len"], value)}
                />
              </label>
              <label className="fieldLabel">
                <span>Maximum training steps</span>
                <ConfigNumberInput
                  value={asNumber(trainingConfig.max_steps, 0)}
                  onCommit={handleMaxStepsChange}
                />
              </label>
              <label className="fieldLabel">
                <span>Total batch size (tokens)</span>
                <ConfigNumberInput
                  value={asNumber(trainingConfig.total_batch_size, 0)}
                  onCommit={(value) => handleTrainingField(["total_batch_size"], value)}
                />
              </label>
              <label className="fieldLabel">
                <span>Micro batch size <small>optional</small></span>
                <OptionalConfigNumberInput
                  value={
                    typeof trainingConfig.micro_batch_size === "number"
                      ? trainingConfig.micro_batch_size
                      : null
                  }
                  onCommit={(value) => handleOptionalTrainingField(["micro_batch_size"], value)}
                  placeholder="Auto"
                />
              </label>
              <label className="fieldLabel">
                <span>Learning rate</span>
                <ConfigNumberInput
                  mode="scientific"
                  step="any"
                  value={asNumber(asRecord(trainingConfig.optimizer).lr, 0.0003)}
                  onCommit={(value) => handleTrainingField(["optimizer", "lr"], value)}
                />
              </label>
              <label className="fieldLabel">
                <span>Weight decay</span>
                <ConfigNumberInput
                  mode="decimal"
                  step="0.0001"
                  value={asNumber(asRecord(trainingConfig.optimizer).weight_decay, 0.1)}
                  onCommit={(value) => handleTrainingField(["optimizer", "weight_decay"], value)}
                />
              </label>
              <label className="fieldLabel">
                <span>Save checkpoint every</span>
                <ConfigNumberInput
                  value={asNumber(trainingConfig.save_every, 0)}
                  onCommit={(value) => handleTrainingField(["save_every"], value)}
                />
              </label>
              <label className="fieldLabel">
                <span>Generate samples every</span>
                <ConfigNumberInput
                  value={asNumber(trainingConfig.sample_every, 0)}
                  onCommit={(value) => handleTrainingField(["sample_every"], value)}
                />
              </label>
              <label className="fieldLabel">
                <span>Dataset shuffle buffer</span>
                <ConfigNumberInput
                  value={asNumber(asRecord(dataloaderConfig.shuffle).buffer_size, 1000)}
                  onCommit={(value) => handleDataloaderField(["shuffle", "buffer_size"], value)}
                />
              </label>
            </div>

            <BatchLrAdvisor
              onApplyRecommendation={onApplyRecommendation}
              onRefreshRecommendation={onRefreshRecommendation}
              preflightError={preflightError}
              preflightLoading={preflightLoading}
              recommendation={recommendation}
              selectedRecommendationOptionKey={selectedRecommendationOptionKey}
              setSelectedRecommendationOptionKey={setSelectedRecommendationOptionKey}
              trainingConfig={trainingConfig}
            />

            <LearningRateSchedulePlanner
              baseLearningRate={asNumber(asRecord(trainingConfig.optimizer).lr, 0.0003)}
              maxSteps={asNumber(trainingConfig.max_steps, 0)}
              schedulerConfig={asRecord(trainingConfig.lr_scheduler)}
              onSchedulersChange={handleLrSchedulersChange}
            />
          </div>
        </div>
      </details>
    );
  }
);
