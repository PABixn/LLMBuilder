import { FiRefreshCw, FiSend } from "react-icons/fi";

import {
  SIMPLE_CREATIVITY_PRESETS,
  SIMPLE_LENGTH_PRESETS,
} from "../lib/inferencePresets";
import type {
  SimpleInferenceCreativity,
  SimpleInferenceLength,
  SimpleModeController,
} from "../types";

interface InferenceStepProps {
  controller: SimpleModeController;
}

export function InferenceStep({ controller }: InferenceStepProps) {
  const { inferenceStep } = controller;
  const canGenerate =
    Boolean(inferenceStep.selectedRun && inferenceStep.latestCheckpoint) &&
    inferenceStep.prompt.trim() !== "" &&
    !inferenceStep.generating;

  return (
    <div className="simpleStepGrid">
      <div className="simplePanel">
        <div className="simpleArtifactList">
          <div>
            <span>Training run</span>
            <strong>{inferenceStep.selectedRun?.name ?? "No completed run"}</strong>
          </div>
          <div>
            <span>Checkpoint</span>
            <strong>
              {inferenceStep.latestCheckpoint
                ? `Latest, step ${inferenceStep.latestCheckpoint.step.toLocaleString()}`
                : "No checkpoint"}
            </strong>
          </div>
        </div>

        <label className="fieldLabel">
          <span>Prompt</span>
          <textarea
            value={inferenceStep.prompt}
            rows={6}
            onChange={(event) => inferenceStep.setPrompt(event.currentTarget.value)}
          />
        </label>

        <div className="simpleOptionRows simpleOptionRowsCompact">
          <div className="simpleOptionLine">
            <span>Length</span>
            <div className="simpleMiniSegmented" role="group" aria-label="Generation length">
              {(Object.keys(SIMPLE_LENGTH_PRESETS) as SimpleInferenceLength[]).map((id) => (
                <button
                  key={id}
                  type="button"
                  className={inferenceStep.lengthPreset === id ? "is-selected" : ""}
                  onClick={() => inferenceStep.setLengthPreset(id)}
                >
                  {SIMPLE_LENGTH_PRESETS[id].label}
                </button>
              ))}
            </div>
          </div>
          <div className="simpleOptionLine">
            <span>Creativity</span>
            <div className="simpleMiniSegmented" role="group" aria-label="Generation creativity">
              {(Object.keys(SIMPLE_CREATIVITY_PRESETS) as SimpleInferenceCreativity[]).map((id) => (
                <button
                  key={id}
                  type="button"
                  className={inferenceStep.creativityPreset === id ? "is-selected" : ""}
                  onClick={() => inferenceStep.setCreativityPreset(id)}
                >
                  {SIMPLE_CREATIVITY_PRESETS[id].label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {inferenceStep.generationError ? (
          <div className="inlineNotice tone-error">{inferenceStep.generationError}</div>
        ) : null}
        {inferenceStep.checkpointError ? (
          <div className="inlineNotice tone-error">{inferenceStep.checkpointError}</div>
        ) : null}

        <div className="simpleActionRow">
          <button
            type="button"
            className="buttonPrimary"
            disabled={!canGenerate}
            onClick={() => void inferenceStep.generate()}
          >
            <FiSend aria-hidden="true" />
            {inferenceStep.generating ? "Generating" : "Generate"}
          </button>
          <button
            type="button"
            className="buttonGhost"
            disabled={!inferenceStep.result || inferenceStep.generating}
            onClick={() => void inferenceStep.tryAnother()}
          >
            <FiRefreshCw aria-hidden="true" />
            Try another
          </button>
        </div>
      </div>

      <div className="simplePanel">
        <div className="simplePanelHeader">
          <div>
            <p className="simpleEyebrow">Output</p>
            <h3>Generated text</h3>
          </div>
        </div>
        {inferenceStep.result ? (
          <div className="simpleGeneration">
            <span>{inferenceStep.result.prompt}</span>
            <strong>{inferenceStep.result.completion || " "}</strong>
          </div>
        ) : (
          <p className="simpleMuted">Output appears here after generation.</p>
        )}
      </div>
    </div>
  );
}
