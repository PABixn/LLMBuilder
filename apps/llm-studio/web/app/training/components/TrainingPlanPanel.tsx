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
import { FieldLabelText, InfoTooltip } from "../../shared/components/HelpTooltip";
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
        <summary>
          <span>Training plan</span>
          <InfoTooltip label="Training plan explanation" align="right" width="wide">
            <strong>Training plan</strong>
            <p>
              These values control how much context the model sees, how long it trains,
              how large each optimizer update is, and how often outputs are saved.
            </p>
          </InfoTooltip>
        </summary>
        <div className="settingsGrid">
          <div
            id="settings-training"
            ref={trainingSettingsRef}
            className={`settingsGroup settingsCategoryAnchor ${
              highlighted ? "settingsCategoryAnchor-highlight" : ""
            }`}
          >
            <div className="settingsGroupHeader">
              <h3>Core settings</h3>
              <p className="settingsGroupHint">
                Set the main values for this run. Hover each label for the app-specific effect.
              </p>
            </div>
            <div className="fieldGrid trainingSettingsCompactGrid">
              <label className="fieldLabel">
                <FieldLabelText tooltipLabel="Sequence length explanation" tooltip="How many tokens are fed into the model at once. It must fit within the model context length from Model Studio and strongly affects memory use.">
                  Sequence length
                </FieldLabelText>
                <ConfigNumberInput
                  value={asNumber(trainingConfig.seq_len, 128)}
                  onCommit={(value) => handleTrainingField(["seq_len"], value)}
                />
              </label>
              <label className="fieldLabel">
                <FieldLabelText tooltipLabel="Max training steps explanation" tooltip="The number of optimizer updates before the run stops. The recommendation system estimates this from dataset size, target token budget, and batch size.">
                  Max training steps
                </FieldLabelText>
                <ConfigNumberInput
                  value={asNumber(trainingConfig.max_steps, 0)}
                  onCommit={handleMaxStepsChange}
                />
              </label>
              <label className="fieldLabel">
                <FieldLabelText tooltipLabel="Total batch tokens explanation" tooltip="The full token count used for one optimizer step after gradient accumulation. Larger values smooth training but require more memory or more accumulation steps.">
                  Total batch tokens
                </FieldLabelText>
                <ConfigNumberInput
                  value={asNumber(trainingConfig.total_batch_size, 0)}
                  onCommit={(value) => handleTrainingField(["total_batch_size"], value)}
                />
              </label>
              <label className="fieldLabel">
                <FieldLabelText tooltipLabel="Micro batch size explanation" tooltip="Optional manual per-device batch size. Leave Auto unless you need to override preflight; Auto lets the app choose a size that fits the selected device memory.">
                  Micro batch size <small>optional</small>
                </FieldLabelText>
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
                <FieldLabelText tooltipLabel="Learning rate explanation" tooltip="How strongly each optimizer step changes the model weights. The advisor scales it with batch size so applying recommendations keeps these values aligned.">
                  Learning rate
                </FieldLabelText>
                <ConfigNumberInput
                  mode="scientific"
                  step="any"
                  value={asNumber(asRecord(trainingConfig.optimizer).lr, 0.0003)}
                  onCommit={(value) => handleTrainingField(["optimizer", "lr"], value)}
                />
              </label>
              <label className="fieldLabel">
                <FieldLabelText tooltipLabel="Weight decay explanation" tooltip="Regularization applied by the optimizer. It gently discourages very large weights and can improve generalization on small or mixed datasets.">
                  Weight decay
                </FieldLabelText>
                <ConfigNumberInput
                  mode="decimal"
                  step="0.0001"
                  value={asNumber(asRecord(trainingConfig.optimizer).weight_decay, 0.1)}
                  onCommit={(value) => handleTrainingField(["optimizer", "weight_decay"], value)}
                />
              </label>
              <label className="fieldLabel">
                <FieldLabelText tooltipLabel="Checkpoint interval explanation" tooltip="How many training steps pass between saved checkpoints. Smaller intervals give more restore points and inference snapshots, but use more disk space.">
                  Save checkpoint every
                </FieldLabelText>
                <ConfigNumberInput
                  value={asNumber(trainingConfig.save_every, 0)}
                  onCommit={(value) => handleTrainingField(["save_every"], value)}
                />
              </label>
              <label className="fieldLabel">
                <FieldLabelText tooltipLabel="Sample interval explanation" tooltip="How often the run asks the model to generate from your sampling prompts. Use it to watch qualitative progress without stopping training.">
                  Generate samples every
                </FieldLabelText>
                <ConfigNumberInput
                  value={asNumber(trainingConfig.sample_every, 0)}
                  onCommit={(value) => handleTrainingField(["sample_every"], value)}
                />
              </label>
              <label className="fieldLabel">
                <FieldLabelText tooltipLabel="Shuffle buffer explanation" tooltip="How many dataset records are kept in memory for randomization. Larger buffers mix examples better; smaller buffers use less memory and start faster.">
                  Dataset shuffle buffer
                </FieldLabelText>
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
