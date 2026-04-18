"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  FiActivity,
  FiLayers,
  FiMoon,
  FiPlay,
  FiRefreshCw,
  FiSearch,
  FiSun,
  FiXCircle,
} from "react-icons/fi";

import { useThemeMode } from "../../lib/theme";
import { formatBytes } from "../../lib/workspaceAssets";
import {
  fetchTrainingCheckpoints,
  fetchTrainingJobs,
  generateTrainingCompletion,
  type GenerateTrainingCompletionResponse,
  type TrainingCheckpointEntry,
  type TrainingJob,
} from "../../lib/trainingApi";

const DEFAULT_PROMPT = "Once upon a time";
const PICKER_SEARCH_PLACEHOLDER = "Search training runs by name, identifier, tokenizer, or artifact";
const CHECKPOINT_SEARCH_PLACEHOLDER = "Search checkpoints by step, file, date, or size";
const LATEST_CHECKPOINT_VALUE = "latest";

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

function formatJobMeta(job: TrainingJob): string {
  const pieces = [
    `step ${formatInteger(job.last_step)}`,
    `${formatInteger(job.checkpoint_count)} checkpoints`,
    `finished ${formatDate(job.finished_at)}`,
  ];
  return pieces.join(" | ");
}

function matchesJobQuery(job: TrainingJob, normalizedQuery: string): boolean {
  if (normalizedQuery === "") {
    return true;
  }

  return [
    job.id,
    job.name,
    job.project_name,
    job.tokenizer_name,
    job.artifact_bundle_file,
    job.artifact_dir,
    job.stage,
  ]
    .filter((value): value is string => typeof value === "string")
    .some((value) => value.toLowerCase().includes(normalizedQuery));
}

function checkpointOptionValue(checkpoint: TrainingCheckpointEntry): string {
  return String(checkpoint.step);
}

function formatCheckpointName(checkpoint: TrainingCheckpointEntry): string {
  return `Step ${formatInteger(checkpoint.step)}`;
}

function formatCheckpointMeta(checkpoint: TrainingCheckpointEntry): string {
  const pieces = [
    checkpoint.created_at ? `saved ${formatDate(checkpoint.created_at)}` : "saved time unavailable",
    formatBytes(checkpoint.size_bytes),
    `${formatInteger(checkpoint.files.length)} files`,
  ];
  return pieces.join(" | ");
}

function matchesCheckpointQuery(
  checkpoint: TrainingCheckpointEntry,
  normalizedQuery: string
): boolean {
  if (normalizedQuery === "") {
    return true;
  }

  return [
    String(checkpoint.step),
    checkpoint.directory,
    checkpoint.created_at ?? "",
    String(checkpoint.size_bytes),
    ...checkpoint.files,
  ].some((value) => value.toLowerCase().includes(normalizedQuery));
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
  const [pickerOpen, setPickerOpen] = useState(false);
  const [pickerQuery, setPickerQuery] = useState("");
  const [checkpoints, setCheckpoints] = useState<TrainingCheckpointEntry[]>([]);
  const [checkpointValue, setCheckpointValue] = useState(LATEST_CHECKPOINT_VALUE);
  const [checkpointPickerOpen, setCheckpointPickerOpen] = useState(false);
  const [checkpointPickerQuery, setCheckpointPickerQuery] = useState("");
  const [checkpointsLoading, setCheckpointsLoading] = useState(false);
  const [checkpointError, setCheckpointError] = useState<string | null>(null);
  const checkpointRequestIdRef = useRef(0);

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
  const visiblePickerJobs = useMemo(() => {
    const normalizedQuery = pickerQuery.trim().toLowerCase();
    return completedJobs.filter((job) => matchesJobQuery(job, normalizedQuery));
  }, [completedJobs, pickerQuery]);
  const latestCheckpoint = useMemo(
    () => checkpoints.reduce<TrainingCheckpointEntry | null>(
      (latest, checkpoint) =>
        latest === null || checkpoint.step > latest.step ? checkpoint : latest,
      null
    ),
    [checkpoints]
  );
  const selectedCheckpoint = useMemo(() => {
    if (checkpointValue === LATEST_CHECKPOINT_VALUE) {
      return latestCheckpoint;
    }
    const selectedStep = Number(checkpointValue);
    return checkpoints.find((checkpoint) => checkpoint.step === selectedStep) ?? null;
  }, [checkpointValue, checkpoints, latestCheckpoint]);
  const visibleCheckpointOptions = useMemo(() => {
    const normalizedQuery = checkpointPickerQuery.trim().toLowerCase();
    return checkpoints.filter((checkpoint) =>
      matchesCheckpointQuery(checkpoint, normalizedQuery)
    );
  }, [checkpointPickerQuery, checkpoints]);
  const showLatestCheckpointOption = useMemo(() => {
    const normalizedQuery = checkpointPickerQuery.trim().toLowerCase();
    return (
      normalizedQuery === "" ||
      "latest".includes(normalizedQuery) ||
      "newest".includes(normalizedQuery) ||
      "automatic".includes(normalizedQuery)
    );
  }, [checkpointPickerQuery]);
  const generationCheckpointStep =
    checkpointValue === LATEST_CHECKPOINT_VALUE ? null : selectedCheckpoint?.step ?? null;

  const closePicker = useCallback(() => {
    setPickerOpen(false);
    setPickerQuery("");
  }, []);
  const closeCheckpointPicker = useCallback(() => {
    setCheckpointPickerOpen(false);
    setCheckpointPickerQuery("");
  }, []);

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

  useEffect(() => {
    const requestId = checkpointRequestIdRef.current + 1;
    checkpointRequestIdRef.current = requestId;
    setCheckpointValue(LATEST_CHECKPOINT_VALUE);
    setCheckpoints([]);
    setCheckpointError(null);

    if (!selectedJobId) {
      setCheckpointsLoading(false);
      return;
    }

    setCheckpointsLoading(true);
    void fetchTrainingCheckpoints(selectedJobId)
      .then((nextCheckpoints) => {
        if (checkpointRequestIdRef.current !== requestId) {
          return;
        }
        setCheckpoints(nextCheckpoints);
      })
      .catch((caught) => {
        if (checkpointRequestIdRef.current !== requestId) {
          return;
        }
        setCheckpointError(caught instanceof Error ? caught.message : "Failed to load checkpoints.");
      })
      .finally(() => {
        if (checkpointRequestIdRef.current === requestId) {
          setCheckpointsLoading(false);
        }
      });
  }, [selectedJobId]);

  useEffect(() => {
    if (!pickerOpen && !checkpointPickerOpen) {
      return;
    }

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        closePicker();
        closeCheckpointPicker();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [checkpointPickerOpen, closeCheckpointPicker, closePicker, pickerOpen]);

  function handleGenerate() {
    if (!selectedJob || prompt.trim() === "") {
      return;
    }

    setError(null);
    setResult(null);
    setGenerating(true);
    void generateTrainingCompletion(selectedJob.id, {
      prompt,
      checkpoint_step: generationCheckpointStep,
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

      {error ? <div className="inlineNotice tone-error">Inference issue: {error}</div> : null}

      <section className="inferenceLayout">
        <div className="panelCard heroCard inferenceArtifactPanel">
          <div className="panelHead">
            <div>
              <h2>Training Artifact</h2>
              <p className="panelCopy">
                Choose a completed training run and the checkpoint that inference should load.
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

          <div className="inferenceAssetStack">
            {selectedJob ? (
              <div className="trainingAssetCard inferenceModelCard">
                <span className="trainingAssetLabel">Model Artifact</span>
                <span className="trainingAssetName">{completedArtifactName(selectedJob)}</span>
                <span className="trainingAssetMeta">{formatJobMeta(selectedJob)}</span>
                <span className="trainingAssetMeta">Tokenizer: {selectedJob.tokenizer_name}</span>
                {selectedJob.artifact_bundle_file ? (
                  <span className="trainingAssetMeta">{selectedJob.artifact_bundle_file}</span>
                ) : null}
                <div className="trainingAssetActions">
                  <button
                    type="button"
                    className="buttonGhost buttonSmall"
                    aria-haspopup="dialog"
                    aria-expanded={pickerOpen}
                    onClick={() => setPickerOpen(true)}
                  >
                    <FiSearch /> Change model artifact
                  </button>
                </div>
              </div>
            ) : (
              <div className="trainingAssetCard inferenceModelCard">
                <span className="trainingAssetLabel">Model Artifact</span>
                <span className="trainingAssetName">No model selected</span>
                <span className="trainingAssetMeta">
                  Finish a model-training run with checkpoints before using inference.
                </span>
                <div className="trainingAssetActions">
                  <button
                    type="button"
                    className="buttonGhost buttonSmall"
                    aria-haspopup="dialog"
                    aria-expanded={pickerOpen}
                    disabled={loading}
                    onClick={() => setPickerOpen(true)}
                  >
                    <FiSearch /> Choose model artifact
                  </button>
                </div>
              </div>
            )}

            <div className="trainingAssetCard inferenceCheckpointCard">
              <span className="trainingAssetLabel">Checkpoint</span>
              <span className="trainingAssetName">
                {checkpointValue === LATEST_CHECKPOINT_VALUE
                  ? "Latest checkpoint"
                  : selectedCheckpoint
                    ? formatCheckpointName(selectedCheckpoint)
                    : "No checkpoint selected"}
              </span>
              <span className="trainingAssetMeta">
                {checkpointsLoading
                  ? "Loading checkpoints..."
                  : checkpointError
                    ? checkpointError
                    : selectedCheckpoint
                      ? formatCheckpointMeta(selectedCheckpoint)
                      : selectedJob
                        ? "No checkpoints are available for this run."
                        : "Choose a model artifact first."}
              </span>
              {checkpointValue === LATEST_CHECKPOINT_VALUE && selectedCheckpoint ? (
                <span className="trainingAssetMeta">
                  Currently resolves to {formatCheckpointName(selectedCheckpoint)}.
                </span>
              ) : null}
              {selectedCheckpoint?.files.length ? (
                <span className="trainingAssetMeta">{selectedCheckpoint.files.join(", ")}</span>
              ) : null}
              <div className="trainingAssetActions">
                <button
                  type="button"
                  className="buttonGhost buttonSmall"
                  aria-haspopup="dialog"
                  aria-expanded={checkpointPickerOpen}
                  disabled={!selectedJob || checkpointsLoading}
                  onClick={() => setCheckpointPickerOpen(true)}
                >
                  <FiSearch /> Choose checkpoint
                </button>
              </div>
            </div>
          </div>
        </div>

        <form
          className="panelCard heroCard inferenceComposer"
          onSubmit={(event) => {
            event.preventDefault();
            handleGenerate();
          }}
        >
          <div className="panelHead">
            <div>
              <h2>Autocompletion</h2>
              <p className="panelCopy">
                The backend encodes the prefix, loads the selected checkpoint, and calls
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

      {pickerOpen ? (
        <div
          className="trainingAssetPickerOverlay"
          onClick={closePicker}
          role="presentation"
        >
          <section
            id="inference-model-picker"
            className="panelCard trainingAssetPickerDialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="inference-model-picker-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="trainingAssetPickerHeader">
              <div>
                <h2 id="inference-model-picker-title">Choose model artifact</h2>
                <p className="panelCopy">
                  Select a completed model-training run. Only runs with saved checkpoints are shown here.
                </p>
              </div>
              <button
                type="button"
                className="buttonGhost iconOnly"
                onClick={closePicker}
                aria-label="Close model artifact picker"
              >
                <FiXCircle />
              </button>
            </div>

            <div className="trainingAssetPickerControls">
              <label className="trainingAssetPickerSearch">
                <FiSearch />
                <input
                  value={pickerQuery}
                  onChange={(event) => setPickerQuery(event.target.value)}
                  placeholder={PICKER_SEARCH_PLACEHOLDER}
                />
              </label>
              <button
                type="button"
                className="buttonGhost"
                onClick={() => {
                  void refreshJobs();
                }}
                disabled={loading}
              >
                <FiRefreshCw /> Refresh
              </button>
            </div>

            <div className="trainingAssetPickerResults">
              {loading ? <div className="trainingEmpty">Loading training artifacts...</div> : null}

              {!loading && error ? (
                <div className="inlineNotice tone-info">{error}</div>
              ) : null}

              {!loading && !error && completedJobs.length === 0 ? (
                <div className="trainingAssetPickerEmpty">
                  <h3>No completed model artifacts found.</h3>
                  <p className="panelCopy">
                    Finish a training run with at least one saved checkpoint, then reopen the picker.
                  </p>
                  <Link className="buttonGhost" href="/training">
                    <FiActivity /> Open Training
                  </Link>
                </div>
              ) : null}

              {!loading && !error && completedJobs.length > 0 && visiblePickerJobs.length === 0 ? (
                <div className="trainingAssetPickerEmpty">
                  <h3>No matching model artifacts.</h3>
                  <p className="panelCopy">
                    Clear the search or refresh the list to check for newly completed runs.
                  </p>
                  <button type="button" className="buttonGhost" onClick={() => setPickerQuery("")}>
                    <FiLayers /> Clear search
                  </button>
                </div>
              ) : null}

              {!loading && !error
                ? visiblePickerJobs.map((job) => (
                    <button
                      key={job.id}
                      type="button"
                      className={`trainingAssetPickerOption ${
                        selectedJobId === job.id ? "is-selected" : ""
                      }`}
                      onClick={() => {
                        setSelectedJobId(job.id);
                        closePicker();
                      }}
                    >
                      <div className="trainingAssetPickerOptionHead">
                        <div>
                          <div className="trainingAssetName">{completedArtifactName(job)}</div>
                        </div>
                        <span
                          className={`pillBadge ${
                            selectedJobId === job.id ? "tone-good" : "tone-neutral"
                          }`}
                        >
                          {selectedJobId === job.id ? "Selected" : "Use model"}
                        </span>
                      </div>
                      <div className="trainingAssetPickerOptionMeta">{formatJobMeta(job)}</div>
                      <div className="trainingAssetPickerOptionMeta">
                        Tokenizer: {job.tokenizer_name}
                      </div>
                      <div className="trainingAssetPickerOptionMeta">
                        {job.artifact_bundle_file ?? job.artifact_dir}
                      </div>
                    </button>
                  ))
                : null}
            </div>
          </section>
        </div>
      ) : null}

      {checkpointPickerOpen ? (
        <div
          className="trainingAssetPickerOverlay"
          onClick={closeCheckpointPicker}
          role="presentation"
        >
          <section
            id="inference-checkpoint-picker"
            className="panelCard trainingAssetPickerDialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="inference-checkpoint-picker-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="trainingAssetPickerHeader">
              <div>
                <h2 id="inference-checkpoint-picker-title">Choose checkpoint</h2>
                <p className="panelCopy">
                  Use the latest checkpoint automatically, or pin inference to a specific saved step from this run.
                </p>
              </div>
              <button
                type="button"
                className="buttonGhost iconOnly"
                onClick={closeCheckpointPicker}
                aria-label="Close checkpoint picker"
              >
                <FiXCircle />
              </button>
            </div>

            <div className="trainingAssetPickerControls">
              <label className="trainingAssetPickerSearch">
                <FiSearch />
                <input
                  value={checkpointPickerQuery}
                  onChange={(event) => setCheckpointPickerQuery(event.target.value)}
                  placeholder={CHECKPOINT_SEARCH_PLACEHOLDER}
                />
              </label>
              <button
                type="button"
                className="buttonGhost"
                disabled={!selectedJob || checkpointsLoading}
                onClick={() => {
                  if (!selectedJob) {
                    return;
                  }
                  const requestId = checkpointRequestIdRef.current + 1;
                  checkpointRequestIdRef.current = requestId;
                  setCheckpointsLoading(true);
                  setCheckpointError(null);
                  void fetchTrainingCheckpoints(selectedJob.id)
                    .then((nextCheckpoints) => {
                      if (checkpointRequestIdRef.current !== requestId) {
                        return;
                      }
                      setCheckpoints(nextCheckpoints);
                    })
                    .catch((caught) => {
                      if (checkpointRequestIdRef.current !== requestId) {
                        return;
                      }
                      setCheckpointError(caught instanceof Error ? caught.message : "Failed to load checkpoints.");
                    })
                    .finally(() => {
                      if (checkpointRequestIdRef.current === requestId) {
                        setCheckpointsLoading(false);
                      }
                    });
                }}
              >
                <FiRefreshCw /> Refresh
              </button>
            </div>

            <div className="trainingAssetPickerResults">
              {checkpointsLoading ? <div className="trainingEmpty">Loading checkpoints...</div> : null}

              {!checkpointsLoading && checkpointError ? (
                <div className="inlineNotice tone-info">{checkpointError}</div>
              ) : null}

              {!checkpointsLoading && !checkpointError && checkpoints.length === 0 ? (
                <div className="trainingAssetPickerEmpty">
                  <h3>No checkpoints found for this run.</h3>
                  <p className="panelCopy">
                    Select a completed training run that saved checkpoint files during training.
                  </p>
                  <Link className="buttonGhost" href="/training">
                    <FiActivity /> Open Training
                  </Link>
                </div>
              ) : null}

              {!checkpointsLoading && !checkpointError && checkpoints.length > 0 ? (
                <>
                  {showLatestCheckpointOption ? (
                    <button
                      type="button"
                      className={`trainingAssetPickerOption ${
                        checkpointValue === LATEST_CHECKPOINT_VALUE ? "is-selected" : ""
                      }`}
                      onClick={() => {
                        setCheckpointValue(LATEST_CHECKPOINT_VALUE);
                        closeCheckpointPicker();
                      }}
                    >
                      <div className="trainingAssetPickerOptionHead">
                        <div>
                          <div className="trainingAssetName">Latest checkpoint</div>
                        </div>
                        <span
                          className={`pillBadge ${
                            checkpointValue === LATEST_CHECKPOINT_VALUE ? "tone-good" : "tone-neutral"
                          }`}
                        >
                          {checkpointValue === LATEST_CHECKPOINT_VALUE ? "Selected" : "Use latest"}
                        </span>
                      </div>
                      <div className="trainingAssetPickerOptionMeta">
                        Automatically resolves to the highest saved step when generation starts.
                      </div>
                      {latestCheckpoint ? (
                        <div className="trainingAssetPickerOptionMeta">
                          Current latest: {formatCheckpointName(latestCheckpoint)} | {formatCheckpointMeta(latestCheckpoint)}
                        </div>
                      ) : null}
                    </button>
                  ) : null}

                  {!showLatestCheckpointOption && visibleCheckpointOptions.length === 0 ? (
                    <div className="trainingAssetPickerEmpty">
                      <h3>No matching checkpoints.</h3>
                      <p className="panelCopy">
                        Clear the search to view every saved checkpoint for this run.
                      </p>
                      <button type="button" className="buttonGhost" onClick={() => setCheckpointPickerQuery("")}>
                        <FiLayers /> Clear search
                      </button>
                    </div>
                  ) : null}

                  {visibleCheckpointOptions.map((checkpoint) => {
                    const value = checkpointOptionValue(checkpoint);
                    return (
                      <button
                        key={checkpoint.directory}
                        type="button"
                        className={`trainingAssetPickerOption ${
                          checkpointValue === value ? "is-selected" : ""
                        }`}
                        onClick={() => {
                          setCheckpointValue(value);
                          closeCheckpointPicker();
                        }}
                      >
                        <div className="trainingAssetPickerOptionHead">
                          <div>
                            <div className="trainingAssetName">{formatCheckpointName(checkpoint)}</div>
                          </div>
                          <span
                            className={`pillBadge ${
                              checkpointValue === value ? "tone-good" : "tone-neutral"
                            }`}
                          >
                            {checkpointValue === value ? "Selected" : "Use checkpoint"}
                          </span>
                        </div>
                        <div className="trainingAssetPickerOptionMeta">
                          {formatCheckpointMeta(checkpoint)}
                        </div>
                        <div className="trainingAssetPickerOptionMeta">
                          {checkpoint.files.join(", ") || checkpoint.directory}
                        </div>
                      </button>
                    );
                  })}
                </>
              ) : null}
            </div>
          </section>
        </div>
      ) : null}
    </main>
  );
}
