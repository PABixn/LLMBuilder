import { FiExternalLink, FiPlay, FiRefreshCw } from "react-icons/fi";

import { readTokenizerVocabSize } from "../lib/vocabularySync";
import type {
  SimpleExecutionKind,
  SimpleModeController,
  SimpleTrainingProfile,
} from "../types";
import { SimpleRunMonitor } from "./SimpleRunMonitor";

interface TrainingStepProps {
  controller: SimpleModeController;
}

const PROFILE_OPTIONS: Array<{ id: SimpleTrainingProfile; label: string; description: string }> = [
  { id: "quick", label: "Quick check", description: "20-100 steps for fast feedback" },
  { id: "balanced", label: "Balanced", description: "Recommended batch and learning rate" },
  { id: "longer", label: "Longer run", description: "Conservative capped extension" },
];

const EXECUTION_OPTIONS: Array<{ id: SimpleExecutionKind; label: string; description: string }> = [
  { id: "local", label: "Local machine", description: "No cloud launch" },
  { id: "runpod_pod", label: "RunPod", description: "GPU pod with confirmation" },
];

function readNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function readOptimizerLearningRate(config: Record<string, unknown> | null): number | null {
  if (
    !config ||
    typeof config.optimizer !== "object" ||
    config.optimizer === null ||
    Array.isArray(config.optimizer)
  ) {
    return null;
  }
  return readNumber((config.optimizer as Record<string, unknown>).lr);
}

function formatInteger(value: number | null): string {
  return value === null ? "Loading" : Math.trunc(value).toLocaleString();
}

function formatLearningRate(value: number | null): string {
  return value === null ? "Loading" : value.toExponential(1);
}

export function TrainingStep({ controller }: TrainingStepProps) {
  const { flow, modelStep, tokenizerStep, trainingStep, updateFlow } = controller;
  const tokenizerVocabSize = readTokenizerVocabSize(tokenizerStep.tokenizerJob);
  const trainingConfig = trainingStep.trainingConfig;
  const seqLen = readNumber(trainingConfig?.seq_len);
  const maxSteps = readNumber(trainingConfig?.max_steps);
  const totalBatchSize = readNumber(trainingConfig?.total_batch_size);
  const learningRate = readOptimizerLearningRate(trainingConfig);

  return (
    <div className="simpleStepGrid">
      <div className="simplePanel">
        <div className="simplePanelHeader">
          <div>
            <p className="simpleEyebrow">Plan</p>
            <h3>Recommended training</h3>
          </div>
          <a className="buttonGhost buttonSmall" href="/training">
            <FiExternalLink aria-hidden="true" /> Expert
          </a>
        </div>

        <div className="simpleArtifactList">
          <div>
            <span>Model</span>
            <strong>{modelStep.project?.name ?? flow.modelName}</strong>
          </div>
          <div>
            <span>Tokenizer</span>
            <strong>
              {String(tokenizerStep.tokenizerJob?.tokenizer_config.name ?? "Tokenizer pending")}
            </strong>
          </div>
          <div>
            <span>Vocabulary</span>
            <strong>
              {tokenizerVocabSize
                ? tokenizerVocabSize.toLocaleString()
                : flow.targetVocabSize.toLocaleString()}
            </strong>
          </div>
        </div>

        <div className="simpleSegmented">
          {PROFILE_OPTIONS.map((option) => (
            <button
              key={option.id}
              type="button"
              className={flow.trainingProfile === option.id ? "is-selected" : ""}
              onClick={() =>
                updateFlow((current) => ({
                  ...current,
                  trainingProfile: option.id,
                }))
              }
            >
              <strong>{option.label}</strong>
              <span>{option.description}</span>
            </button>
          ))}
        </div>

        <div className="simpleSummaryGrid simpleSummaryGridFour">
          <span>
            <strong>{formatInteger(seqLen)}</strong>
            <small>Sequence length</small>
          </span>
          <span>
            <strong>{formatInteger(maxSteps)}</strong>
            <small>Max steps</small>
          </span>
          <span>
            <strong>{formatInteger(totalBatchSize)}</strong>
            <small>Total batch tokens</small>
          </span>
          <span>
            <strong>{formatLearningRate(learningRate)}</strong>
            <small>Learning rate</small>
          </span>
        </div>

        <div className="simpleSegmented compact">
          {EXECUTION_OPTIONS.map((option) => (
            <button
              key={option.id}
              type="button"
              className={flow.executionKind === option.id ? "is-selected" : ""}
              onClick={() =>
                updateFlow((current) => ({
                  ...current,
                  executionKind: option.id,
                }))
              }
            >
              <strong>{option.label}</strong>
              <span>{option.description}</span>
            </button>
          ))}
        </div>

        {flow.executionKind === "runpod_pod" ? (
          <label className="simpleCheckbox">
            <input
              type="checkbox"
              checked={trainingStep.cloudConfirmed}
              onChange={(event) => trainingStep.setCloudConfirmed(event.currentTarget.checked)}
            />
            <span>Confirm cloud execution and configured cleanup policy.</span>
          </label>
        ) : null}

        <div className="inlineNotice tone-info">{trainingStep.profileNote}</div>
        {trainingStep.appliedFixes.length > 0 ? (
          <div className="inlineNotice tone-success">
            Applied: {trainingStep.appliedFixes.join(", ")}
          </div>
        ) : null}
      </div>

      <div className="simplePanel">
        <div className="simplePanelHeader">
          <div>
            <p className="simpleEyebrow">Launch</p>
            <h3>Run preflight</h3>
          </div>
          {trainingStep.preflight?.valid ? (
            <span className="simpleStatusPill is-completed">Passed</span>
          ) : null}
        </div>

        {trainingStep.preflight?.warnings.length ? (
          <div className="inlineNotice tone-info">
            {trainingStep.preflight.warnings[0].message}
          </div>
        ) : null}
        {trainingStep.preflightError ? (
          <div className="inlineNotice tone-error">{trainingStep.preflightError}</div>
        ) : null}
        {trainingStep.preflight && !trainingStep.preflight.valid && trainingStep.preflight.errors[0] ? (
          <div className="inlineNotice tone-error">{trainingStep.preflight.errors[0].message}</div>
        ) : null}

        <div className="simpleActionRow">
          <button
            type="button"
            className="buttonGhost"
            disabled={trainingStep.preflightLoading}
            onClick={trainingStep.runPreflight}
          >
            <FiRefreshCw aria-hidden="true" />
            {trainingStep.preflightLoading ? "Checking" : "Run preflight"}
          </button>
          <button
            type="button"
            className="buttonPrimary"
            disabled={!trainingStep.preflight?.valid || trainingStep.launching}
            onClick={() => void trainingStep.startTraining()}
          >
            <FiPlay aria-hidden="true" />
            {trainingStep.launching ? "Starting" : "Start training"}
          </button>
        </div>

        <SimpleRunMonitor
          run={trainingStep.trainingRun}
          metrics={trainingStep.metrics}
          checkpointCount={trainingStep.checkpoints.length}
          sampleCount={trainingStep.samples.length}
        />
      </div>
    </div>
  );
}
