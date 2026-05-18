import type { TrainingSampleEntry } from "../../../lib/training/types";
import { formatInteger } from "../lib/metrics";
import {
  samplePromptSummary,
  splitGeneratedSampleText,
} from "../lib/run";

interface SamplesPanelProps {
  samples: TrainingSampleEntry[];
}

export function SamplesPanel({ samples }: SamplesPanelProps) {
  return (
    <details className="sectionDisclosure">
      <summary className="sectionDisclosureSummary">Samples</summary>
      <div className="trainingSampleList">
        {samples.length ? (
          samples.slice().reverse().map((entry) => {
            const sampleCount = entry.samples.length;
            const totalChars = entry.samples.reduce(
              (sum, sample) => sum + sample.text.length + (sample.prompt?.length ?? 0),
              0
            );

            return (
              <details
                key={`sample-${entry.step}`}
                className="trainingSampleCard trainingSampleStepDisclosure"
              >
                <summary className="trainingSampleStepSummary">
                  <span>
                    <span className="trainingSampleTitle">Step {entry.step}</span>
                    <span className="trainingSampleMeta">
                      {sampleCount} prompt{sampleCount === 1 ? "" : "s"} generated
                      {" - "}
                      {formatInteger(totalChars)} chars
                    </span>
                  </span>
                </summary>

                <div className="trainingSampleStepBody">
                  {entry.samples.map((sample) => {
                    const promptSummary = samplePromptSummary(sample.prompt, sample.index);
                    const splitSample = splitGeneratedSampleText(sample.text, sample.prompt);
                    const continuationLength = splitSample.continuation.length;

                    return (
                      <details
                        key={`${entry.step}-${sample.index}`}
                        className="trainingSampleTextDisclosure"
                      >
                        <summary className="trainingSampleTextSummary">
                          <span className="trainingSamplePromptSummary">{promptSummary}</span>
                          <span className="trainingSampleMeta">
                            {formatInteger(continuationLength)} continuation chars
                          </span>
                        </summary>
                        <div className="trainingSampleGeneratedBlock">
                          <div className="trainingSampleGeneratedHead">
                            <span>Generated text</span>
                            <span>{formatInteger(sample.text.length)} chars</span>
                          </div>
                          <pre className="trainingSampleGeneratedText">
                            {splitSample.prefix ? (
                              <>
                                <span className="trainingSampleGeneratedPrompt">{splitSample.prefix}</span>
                                <span className="trainingSampleGeneratedContinuation">{splitSample.continuation}</span>
                              </>
                            ) : (
                              <span className="trainingSampleGeneratedContinuation">{splitSample.continuation}</span>
                            )}
                          </pre>
                        </div>
                      </details>
                    );
                  })}
                </div>
              </details>
            );
          })
        ) : (
          <div className="trainingEmpty">No samples yet.</div>
        )}
      </div>
    </details>
  );
}
