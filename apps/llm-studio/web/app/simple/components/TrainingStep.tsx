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
  { id: "quick", label: "Quick check", description: "Short local smoke test" },
  { id: "balanced", label: "Balanced", description: "Backend recommendation" },
  { id: "longer", label: "Longer run", description: "Conservative extension" },
];

const EXECUTION_OPTIONS: Array<{ id: SimpleExecutionKind; label: string }> = [
  { id: "local", label: "Local machine" },
  { id: "runpod_pod", label: "RunPod" },
];

export function TrainingStep({ controller }: TrainingStepProps) {
  const { flow, modelStep, tokenizerStep, trainingStep, updateFlow } = controller;
  const tokenizerVocabSize = readTokenizerVocabSize(tokenizerStep.tokenizerJob);

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

        <details className="simpleDetails">
          <summary>Advanced details</summary>
          <pre>{JSON.stringify({ training: trainingStep.trainingConfig, dataloader: trainingStep.dataloaderConfig }, null, 2)}</pre>
        </details>
      </div>
    </div>
  );
}
