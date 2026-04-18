"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  FiActivity,
  FiArchive,
  FiCpu,
  FiMoon,
  FiPlay,
  FiRefreshCw,
  FiSun,
  FiZap,
} from "react-icons/fi";

import { useThemeMode } from "../../lib/theme";
import {
  fetchTrainingJobs,
  generateTrainingCompletion,
  type GenerateTrainingCompletionResponse,
  type TrainingJob,
} from "../../lib/trainingApi";

const DEFAULT_PROMPT = "Once upon a time";

function formatInteger(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "0";
  }
  return Math.round(value).toLocaleString();
}

function formatDate(value: string | null): string {
  if (!value) {
    return "not finished";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}

function completedArtifactName(job: TrainingJob): string {
  const name = job.name?.trim();
  if (name) {
    return name;
  }
  return job.artifact_bundle_file ?? `Training run ${job.id.slice(0, 8)}`;
}

export default function InferencePage() {
  const [theme, setTheme] = useThemeMode();
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [selectedJobId, setSelectedJobId] = useState("");
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [maxTokens, setMaxTokens] = useState(64);
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(50);
  const [seed, setSeed] = useState(42);
  const [repetitionPenalty, setRepetitionPenalty] = useState(1);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<GenerateTrainingCompletionResponse | null>(null);

  const completedJobs = useMemo(
    () =>
      jobs.filter(
        (job) => job.status === "completed" && job.checkpoint_count > 0
      ),
    [jobs]
  );
  const selectedJob = useMemo(
    () => completedJobs.find((job) => job.id === selectedJobId) ?? null,
    [completedJobs, selectedJobId]
  );

  const refreshJobs = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const nextJobs = await fetchTrainingJobs();
      setJobs(nextJobs);
      const nextCompletedJobs = nextJobs.filter(
        (job) => job.status === "completed" && job.checkpoint_count > 0
      );
      setSelectedJobId((current) => {
        if (current && nextCompletedJobs.some((job) => job.id === current)) {
          return current;
        }
        return nextCompletedJobs[0]?.id ?? "";
      });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Failed to load training artifacts.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refreshJobs();
  }, [refreshJobs]);

  function handleGenerate() {
    if (!selectedJob || prompt.trim() === "") {
      return;
    }

    setError(null);
    setResult(null);
    setGenerating(true);
    void generateTrainingCompletion(selectedJob.id, {
      prompt,
      max_tokens: maxTokens,
      temperature,
      top_k: topK,
      seed,
      repetition_penalty: repetitionPenalty,
    })
      .then((response) => {
        setResult(response);
      })
      .catch((caught) => {
        setError(caught instanceof Error ? caught.message : "Generation failed.");
      })
      .finally(() => {
        setGenerating(false);
      });
  }

  return (
    <main className="studioRoot inferencePage">
      <header className="studioNav" role="navigation" aria-label="Primary">
        <div className="studioNavBrand">
          <span className="studioNavDot" />
          <span>LLM Builder</span>
        </div>
        <nav className="studioNavLinks" aria-label="Primary routes">
          <Link className="studioNavLink" href="/">
            Home
          </Link>
          <Link className="studioNavLink" href="/studio">
            LLM Studio
          </Link>
          <Link className="studioNavLink" href="/tokenizer">
            Tokenizer Studio
          </Link>
          <Link className="studioNavLink" href="/training">
            Training
          </Link>
          <Link className="studioNavLink" href="/inference" aria-current="page">
            Inference
          </Link>
        </nav>
        <button
          className="themeToggle"
          onClick={() => setTheme((previous) => (previous === "dark" ? "white" : "dark"))}
          aria-label="Toggle theme"
        >
          {theme === "dark" ? <FiSun /> : <FiMoon />}
        </button>
      </header>

      <section className="panelCard inferenceHero">
        <div className="panelHead heroHead">
          <div>
            <h1>Autocomplete from a trained artifact.</h1>
            <p className="panelCopy">
              Select one completed training run, enter a text prefix, and generate a plain
              continuation from the latest checkpoint.
            </p>
          </div>
          <div className="actionCluster">
            <button type="button" className="buttonGhost" onClick={() => void refreshJobs()} disabled={loading}>
              <FiRefreshCw /> Refresh
            </button>
            <Link className="buttonGhost" href="/training">
              <FiActivity /> Training
            </Link>
          </div>
        </div>
        <div className="inferenceStatusGrid">
          <div className="statusCard">
            <div className="statusCardIcon"><FiArchive /></div>
            <div>
              <div className="statusCardTitle">Artifacts</div>
              <div className="statusCardValue">{loading ? "..." : formatInteger(completedJobs.length)}</div>
              <div className="statusCardDetail">Completed runs with checkpoints</div>
            </div>
          </div>
          <div className="statusCard">
            <div className="statusCardIcon"><FiCpu /></div>
            <div>
              <div className="statusCardTitle">Checkpoint</div>
              <div className="statusCardValue">{selectedJob ? formatInteger(selectedJob.last_step) : "0"}</div>
              <div className="statusCardDetail">{selectedJob ? selectedJob.stage : "No artifact selected"}</div>
            </div>
          </div>
          <div className="statusCard">
            <div className="statusCardIcon"><FiZap /></div>
            <div>
              <div className="statusCardTitle">Mode</div>
              <div className="statusCardValue">Completion</div>
              <div className="statusCardDetail">No chat messages or roles</div>
            </div>
          </div>
        </div>
      </section>

      {error ? <div className="inlineNotice tone-error">Inference issue: {error}</div> : null}

      <section className="inferenceLayout">
        <div className="panelCard inferenceArtifactPanel">
          <div className="panelHead">
            <div>
              <h2>Training Artifact</h2>
              <p className="panelCopy">
                Only completed model-training runs with saved checkpoints are available here.
              </p>
            </div>
          </div>

          <label className="fieldLabel">
            <span>Model artifact</span>
            <select value={selectedJobId} onChange={(event) => setSelectedJobId(event.target.value)} disabled={loading || completedJobs.length === 0}>
              {completedJobs.length === 0 ? (
                <option value="">No completed training artifacts</option>
              ) : null}
              {completedJobs.map((job) => (
                <option key={job.id} value={job.id}>
                  {completedArtifactName(job)}
                </option>
              ))}
            </select>
          </label>

          {selectedJob ? (
            <div className="inferenceArtifactMeta">
              <span>Run id: {selectedJob.id}</span>
              <span>Tokenizer: {selectedJob.tokenizer_name}</span>
              <span>Finished: {formatDate(selectedJob.finished_at)}</span>
              <span>Checkpoints: {formatInteger(selectedJob.checkpoint_count)}</span>
            </div>
          ) : (
            <div className="trainingEmpty">
              Finish a training run before using this page.
            </div>
          )}
        </div>

        <form
          className="panelCard inferenceComposer"
          onSubmit={(event) => {
            event.preventDefault();
            handleGenerate();
          }}
        >
          <div className="panelHead">
            <div>
              <h2>Autocompletion</h2>
              <p className="panelCopy">
                The backend encodes the prefix, loads the latest checkpoint, and calls
                `ConfigurableGPT.generate`.
              </p>
            </div>
            <button
              type="submit"
              className="buttonPrimary"
              disabled={!selectedJob || prompt.trim() === "" || generating}
            >
              <FiPlay /> {generating ? "Generating" : "Generate"}
            </button>
          </div>

          <label className="fieldLabel fullWidthField">
            <span>Prefix text</span>
            <textarea
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              placeholder="Start a sentence and let the model continue it."
            />
          </label>

          <div className="fieldGrid compact">
            <label className="fieldLabel">
              <span>Max tokens</span>
              <input
                type="number"
                min={1}
                max={1024}
                value={maxTokens}
                onChange={(event) => setMaxTokens(Number(event.target.value))}
              />
            </label>
            <label className="fieldLabel">
              <span>Temperature</span>
              <input
                type="number"
                min={0}
                max={5}
                step={0.1}
                value={temperature}
                onChange={(event) => setTemperature(Number(event.target.value))}
              />
            </label>
            <label className="fieldLabel">
              <span>Top K</span>
              <input
                type="number"
                min={1}
                max={50000}
                value={topK}
                onChange={(event) => setTopK(Number(event.target.value))}
              />
            </label>
            <label className="fieldLabel">
              <span>Seed</span>
              <input
                type="number"
                min={0}
                value={seed}
                onChange={(event) => setSeed(Number(event.target.value))}
              />
            </label>
            <label className="fieldLabel">
              <span>Repetition penalty</span>
              <input
                type="number"
                min={0.1}
                max={5}
                step={0.1}
                value={repetitionPenalty}
                onChange={(event) => setRepetitionPenalty(Number(event.target.value))}
              />
            </label>
          </div>
        </form>
      </section>

      <section className="panelCard inferenceOutputPanel">
        <div className="panelHead">
          <div>
            <h2>Continuation</h2>
            <p className="panelCopy">
              Generated text appears as a direct continuation of the prefix.
            </p>
          </div>
          {result ? (
            <span className="trainingAssetMeta">
              Step {formatInteger(result.checkpoint_step)} | {formatInteger(result.generated_token_count)} tokens
            </span>
          ) : null}
        </div>
        {result ? (
          <div className="inferenceOutput">
            <span className="inferencePromptText">{result.prompt}</span>
            <span className="inferenceCompletionText">{result.completion}</span>
          </div>
        ) : (
          <div className="trainingEmpty">
            Run a completion to see model output.
          </div>
        )}
      </section>
    </main>
  );
}
