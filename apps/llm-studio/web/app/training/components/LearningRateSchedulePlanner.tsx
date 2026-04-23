"use client";

import { useEffect, useMemo, useState, type CSSProperties } from "react";
import {
  FiArrowDown,
  FiArrowUp,
  FiCopy,
  FiPlus,
  FiRefreshCw,
  FiTrash2,
} from "react-icons/fi";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  LR_CHART_FRAME_INSET,
  LR_CHART_MARGIN,
  LR_CHART_Y_AXIS_WIDTH,
  SCHEDULER_DEFINITIONS,
  SCHEDULER_TYPES,
  asPositiveInteger,
  asPositiveNumber,
  buildLearningRateProjection,
  clamp,
  defaultMilestones,
  defaultSchedulerForType,
  fitSchedulersToMaxSteps,
  formatInteger,
  formatLearningRate,
  formatNumberInputValue,
  lrAxisDomain,
  makeStepDecaySchedule,
  makeWarmupCosineSchedule,
  makeWarmupHoldSchedule,
  normalizeSchedulers,
  parseConfigNumberInput,
  parseMilestonesInput,
  projectionStats,
  sanitizePositiveDecimalInput,
  sanitizePositiveIntegerInput,
  sanitizePositiveScientificInput,
  sanitizeSchedulerRecord,
  serializeScheduler,
  sumSchedulerSteps,
  type ConfigNumberMode,
  type LearningRateSchedulePlannerProps,
  type LearningRateSchedulerConfig,
  type LearningRateSchedulerType,
  type ProjectionPoint,
} from "../lib/learningRateSchedule";

export { fitSchedulersToMaxSteps } from "../lib/learningRateSchedule";

function LearningRateTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: Array<{ payload?: ProjectionPoint }>;
  label?: number | string;
}) {
  const point = payload?.[0]?.payload;
  if (!active || !point) {
    return null;
  }
  return (
    <div className="trainingChartTooltip">
      <span>Step {label} - phase {point.phaseIndex + 1}</span>
      <strong>{point.phaseLabel}: {formatLearningRate(point.lr)}</strong>
    </div>
  );
}

function PlannerNumberInput({
  value,
  onCommit,
  mode = "integer",
  min = 1,
  max,
  step,
  placeholder,
}: {
  value: number;
  onCommit: (value: number) => void;
  mode?: ConfigNumberMode;
  min?: number;
  max?: number;
  step?: string;
  placeholder?: string;
}) {
  const formattedValue = formatNumberInputValue(value, mode);
  const [draft, setDraft] = useState(formattedValue);
  const [focused, setFocused] = useState(false);

  useEffect(() => {
    if (!focused) {
      setDraft(formattedValue);
    }
  }, [focused, formattedValue]);

  const sanitize = mode === "scientific"
    ? sanitizePositiveScientificInput
    : mode === "decimal"
      ? sanitizePositiveDecimalInput
      : sanitizePositiveIntegerInput;
  const inputMode = mode === "integer" ? "numeric" : "decimal";
  const pattern = mode === "scientific"
    ? "[0-9]*[.]?[0-9]*([eE][+-]?[0-9]+)?"
    : mode === "decimal"
      ? "[0-9]*[.]?[0-9]*"
      : "[0-9]*";

  return (
    <input
      type="text"
      inputMode={inputMode}
      pattern={pattern}
      step={step}
      min={min}
      max={max}
      placeholder={placeholder}
      value={draft}
      onFocus={() => setFocused(true)}
      onChange={(event) => {
        setDraft(sanitize(event.currentTarget.value));
      }}
      onBlur={() => {
        setFocused(false);
        const parsed = parseConfigNumberInput(draft, mode);
        if (parsed === null) {
          setDraft(formattedValue);
          return;
        }
        const upperBound = typeof max === "number" ? max : Number.POSITIVE_INFINITY;
        const nextValue = clamp(parsed, min, upperBound);
        onCommit(mode === "integer" ? Math.trunc(nextValue) : nextValue);
        setDraft(formatNumberInputValue(nextValue, mode));
      }}
      onKeyDown={(event) => {
        if (event.key === "Enter") {
          event.currentTarget.blur();
        }
      }}
    />
  );
}

function SchedulerFields({
  scheduler,
  onPatch,
}: {
  scheduler: LearningRateSchedulerConfig;
  onPatch: (patch: Record<string, unknown>) => void;
}) {
  if (scheduler.type === "linear") {
    return (
      <>
        <label className="fieldLabel">
          <span>Start factor</span>
          <PlannerNumberInput
            mode="decimal"
            min={0.000001}
            step="0.01"
            value={asPositiveNumber(scheduler.start_factor, 0.1)}
            onCommit={(value) => onPatch({ start_factor: value })}
          />
        </label>
        <label className="fieldLabel">
          <span>End factor</span>
          <PlannerNumberInput
            mode="decimal"
            min={0.000001}
            step="0.01"
            value={asPositiveNumber(scheduler.end_factor, 1)}
            onCommit={(value) => onPatch({ end_factor: value })}
          />
        </label>
      </>
    );
  }

  if (scheduler.type === "cosine_annealing") {
    return (
      <label className="fieldLabel">
        <span>Minimum LR</span>
        <PlannerNumberInput
          mode="scientific"
          min={0}
          step="any"
          value={asPositiveNumber(scheduler.eta_min, 0, true)}
          onCommit={(value) => onPatch({ eta_min: value })}
        />
      </label>
    );
  }

  if (scheduler.type === "step") {
    return (
      <>
        <label className="fieldLabel">
          <span>Step size</span>
          <PlannerNumberInput
            min={1}
            max={scheduler.steps}
            value={asPositiveInteger(scheduler.step_size, 1)}
            onCommit={(value) => onPatch({ step_size: value })}
          />
        </label>
        <label className="fieldLabel">
          <span>Gamma</span>
          <PlannerNumberInput
            mode="decimal"
            min={0.000001}
            step="0.01"
            value={asPositiveNumber(scheduler.gamma, 0.5)}
            onCommit={(value) => onPatch({ gamma: value })}
          />
        </label>
      </>
    );
  }

  if (scheduler.type === "multistep") {
    const milestones = Array.isArray(scheduler.milestones)
      ? scheduler.milestones.join(", ")
      : "";
    return (
      <>
        <label className="fieldLabel fullWidthField">
          <span>Milestones</span>
          <input
            key={milestones}
            defaultValue={milestones}
            placeholder="50, 100, 150"
            onBlur={(event) => {
              const nextMilestones = parseMilestonesInput(
                event.currentTarget.value,
                scheduler.steps
              );
              const repairedMilestones = nextMilestones.length
                ? nextMilestones
                : defaultMilestones(scheduler.steps);
              onPatch({ milestones: repairedMilestones });
              event.currentTarget.value = repairedMilestones.join(", ");
            }}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.currentTarget.blur();
              }
            }}
          />
          <span className="fieldNote">Use ascending steps inside this phase.</span>
        </label>
        <label className="fieldLabel">
          <span>Gamma</span>
          <PlannerNumberInput
            mode="decimal"
            min={0.000001}
            step="0.01"
            value={asPositiveNumber(scheduler.gamma, 0.5)}
            onCommit={(value) => onPatch({ gamma: value })}
          />
        </label>
      </>
    );
  }

  if (scheduler.type === "exponential") {
    return (
      <label className="fieldLabel">
        <span>Gamma</span>
        <PlannerNumberInput
          mode="decimal"
          min={0.000001}
          step="0.001"
          value={asPositiveNumber(scheduler.gamma, 0.99)}
          onCommit={(value) => onPatch({ gamma: value })}
        />
      </label>
    );
  }

  if (scheduler.type === "cosine_annealing_warm_restarts") {
    return (
      <>
        <label className="fieldLabel">
          <span>First restart length</span>
          <PlannerNumberInput
            min={1}
            value={asPositiveInteger(scheduler.t_0, 1)}
            onCommit={(value) => onPatch({ t_0: value })}
          />
        </label>
        <label className="fieldLabel">
          <span>Cycle multiplier</span>
          <PlannerNumberInput
            min={1}
            value={asPositiveInteger(scheduler.t_mult, 1)}
            onCommit={(value) => onPatch({ t_mult: value })}
          />
        </label>
        <label className="fieldLabel">
          <span>Minimum LR</span>
          <PlannerNumberInput
            mode="scientific"
            min={0}
            step="any"
            value={asPositiveNumber(scheduler.eta_min, 0, true)}
            onCommit={(value) => onPatch({ eta_min: value })}
          />
        </label>
      </>
    );
  }

  return (
    <label className="fieldLabel">
      <span>LR factor</span>
      <PlannerNumberInput
        mode="decimal"
        min={0.000001}
        step="0.01"
        value={asPositiveNumber(scheduler.factor, 1)}
        onCommit={(value) => onPatch({ factor: value })}
      />
    </label>
  );
}

export function LearningRateSchedulePlanner({
  baseLearningRate,
  maxSteps,
  schedulerConfig,
  onSchedulersChange,
}: LearningRateSchedulePlannerProps) {
  const schedulers = useMemo(
    () =>
      normalizeSchedulers(
        schedulerConfig.schedulers,
        maxSteps,
        Number.isFinite(baseLearningRate) && baseLearningRate > 0 ? baseLearningRate : 0.0003
      ),
    [baseLearningRate, maxSteps, schedulerConfig.schedulers]
  );
  const totalPlannedSteps = sumSchedulerSteps(schedulers);
  const exactStepFit = totalPlannedSteps === maxSteps;
  const stepDelta = maxSteps - totalPlannedSteps;
  const projection = useMemo(
    () => buildLearningRateProjection(schedulers, baseLearningRate),
    [baseLearningRate, schedulers]
  );
  const stats = useMemo(() => projectionStats(projection), [projection]);
  const yDomain = useMemo(() => lrAxisDomain(projection), [projection]);
  const xDomainEnd = Math.max(totalPlannedSteps, 1);
  const hasMultiplePhases = schedulers.length > 1;
  const timelineStyle = {
    "--lr-chart-left-gutter": `${
      LR_CHART_FRAME_INSET + LR_CHART_Y_AXIS_WIDTH + LR_CHART_MARGIN.left
    }px`,
    "--lr-chart-right-gutter": `${LR_CHART_FRAME_INSET + LR_CHART_MARGIN.right}px`,
  } as CSSProperties;

  const commitSchedulers = (nextSchedulers: LearningRateSchedulerConfig[]) => {
    onSchedulersChange(nextSchedulers.map(serializeScheduler));
  };

  const updateSchedulerAt = (
    index: number,
    updater: (scheduler: LearningRateSchedulerConfig) => LearningRateSchedulerConfig
  ) => {
    commitSchedulers(
      schedulers.map((scheduler, currentIndex) =>
        currentIndex === index
          ? sanitizeSchedulerRecord(updater(scheduler), scheduler.steps, baseLearningRate)
          : scheduler
      )
    );
  };

  const applySerializedSchedulers = (nextSchedulers: Record<string, unknown>[]) => {
    onSchedulersChange(nextSchedulers);
  };

  const addPhase = () => {
    const remainingSteps = Math.max(1, maxSteps - totalPlannedSteps);
    const nextType: LearningRateSchedulerType =
      schedulers.length === 0 ? "constant" : "cosine_annealing";
    commitSchedulers([
      ...schedulers,
      defaultSchedulerForType(nextType, remainingSteps, baseLearningRate),
    ]);
  };

  const scheduleStatus = exactStepFit
    ? "Exact fit"
    : stepDelta > 0
      ? `${formatInteger(stepDelta)} unplanned step${stepDelta === 1 ? "" : "s"}`
      : `${formatInteger(Math.abs(stepDelta))} extra scheduled step${
          Math.abs(stepDelta) === 1 ? "" : "s"
        }`;

  return (
    <details className="lrPlanner" open aria-labelledby="lr-planner-title">
      <summary className="lrPlannerSummary">
        <div>
          <h3 id="lr-planner-title">Advanced learning-rate schedule</h3>
          <p className="settingsGroupHint">
            Compose backend scheduler phases, keep their step math aligned with the run, and preview
            the predicted learning rate before launch.
          </p>
        </div>
        <div className="lrPlannerToolbar">
          <span className={`pillBadge ${exactStepFit ? "tone-good" : "tone-warn"}`}>
            {scheduleStatus}
          </span>
        </div>
      </summary>

      <div className="lrPlannerContent">
        <div className="lrPlannerSummaryGrid" aria-label="Learning-rate schedule summary">
          <div className="lrPlannerSummaryItem">
            <span>Base LR</span>
            <strong>{formatLearningRate(baseLearningRate)}</strong>
          </div>
          <div className="lrPlannerSummaryItem">
            <span>Planned steps</span>
            <strong>{formatInteger(totalPlannedSteps)} / {formatInteger(maxSteps)}</strong>
          </div>
          <div className="lrPlannerSummaryItem">
            <span>Peak LR</span>
            <strong>{formatLearningRate(stats?.max)}</strong>
          </div>
          <div className="lrPlannerSummaryItem">
            <span>Final LR</span>
            <strong>{formatLearningRate(stats?.final)}</strong>
          </div>
        </div>

        <div className="lrPlannerPresetRow" aria-label="Learning-rate schedule presets">
          <button
            type="button"
            className="secondaryButton"
            onClick={() =>
              applySerializedSchedulers(makeWarmupCosineSchedule(maxSteps, baseLearningRate))
            }
          >
            Warmup + cosine
          </button>
          <button
            type="button"
            className="secondaryButton"
            onClick={() => applySerializedSchedulers(makeWarmupHoldSchedule(maxSteps))}
          >
            Warmup + hold
          </button>
          <button
            type="button"
            className="secondaryButton"
            onClick={() => applySerializedSchedulers(makeStepDecaySchedule(maxSteps))}
          >
            Step decay
          </button>
          <button
            type="button"
            className="secondaryButton"
            onClick={() =>
              applySerializedSchedulers([
                { type: "constant", steps: Math.max(1, Math.trunc(maxSteps || 1)), factor: 1 },
              ])
            }
          >
            Constant LR
          </button>
        </div>

        <div className="lrPlannerBodyGrid">
          <div
            className={
              hasMultiplePhases
                ? "lrPlannerPhaseList"
                : "lrPlannerPhaseList lrPlannerPhaseList-single"
            }
          >
            <div className="lrPlannerSectionHead">
              <div>
                <strong>Scheduler phases</strong>
                <span>{schedulers.length} phase{schedulers.length === 1 ? "" : "s"}</span>
              </div>
              <div className="lrPlannerSectionActions">
                <button
                  type="button"
                  className="secondaryButton"
                  onClick={() =>
                    applySerializedSchedulers(
                      fitSchedulersToMaxSteps(
                        schedulers.map(serializeScheduler),
                        maxSteps,
                        baseLearningRate
                      )
                    )
                  }
                >
                  Fit to max steps
                </button>
                <button type="button" className="secondaryButton" onClick={addPhase}>
                  <FiPlus aria-hidden="true" /> Add phase
                </button>
              </div>
            </div>

            {schedulers.map((scheduler, index) => {
              const definition = SCHEDULER_DEFINITIONS[scheduler.type];
              const phaseStart = schedulers
                .slice(0, index)
                .reduce((sum, item) => sum + item.steps, 0);
              const phaseEnd = phaseStart + scheduler.steps;

              return (
                <article
                  key={`${index}-${scheduler.type}`}
                  className={
                    hasMultiplePhases
                      ? "lrPlannerPhaseCard"
                      : "lrPlannerPhaseCard lrPlannerPhaseCard-single"
                  }
                >
                  <div className="lrPlannerPhaseHeader">
                    <div>
                      <span>Phase {index + 1}</span>
                      <strong>{definition.label}</strong>
                      <p>{definition.bestFor}</p>
                    </div>
                    <div className="lrPlannerPhaseActions">
                      {hasMultiplePhases ? (
                        <>
                          <button
                            type="button"
                            className="trainingRecentIconButton"
                            onClick={() => {
                              if (index === 0) {
                                return;
                              }
                              const next = [...schedulers];
                              [next[index - 1], next[index]] = [next[index], next[index - 1]];
                              commitSchedulers(next);
                            }}
                            disabled={index === 0}
                            aria-label={`Move phase ${index + 1} up`}
                            title="Move up"
                          >
                            <FiArrowUp aria-hidden="true" />
                          </button>
                          <button
                            type="button"
                            className="trainingRecentIconButton"
                            onClick={() => {
                              if (index === schedulers.length - 1) {
                                return;
                              }
                              const next = [...schedulers];
                              [next[index], next[index + 1]] = [next[index + 1], next[index]];
                              commitSchedulers(next);
                            }}
                            disabled={index === schedulers.length - 1}
                            aria-label={`Move phase ${index + 1} down`}
                            title="Move down"
                          >
                            <FiArrowDown aria-hidden="true" />
                          </button>
                        </>
                      ) : null}
                      <button
                        type="button"
                        className={
                          hasMultiplePhases
                            ? "trainingRecentIconButton"
                            : "buttonGhost buttonSmall lrPlannerDuplicatePhaseButton"
                        }
                        onClick={() =>
                          commitSchedulers([
                            ...schedulers.slice(0, index + 1),
                            scheduler,
                            ...schedulers.slice(index + 1),
                          ])
                        }
                        aria-label={`Duplicate phase ${index + 1}`}
                        title={hasMultiplePhases ? "Duplicate" : undefined}
                      >
                        <FiCopy aria-hidden="true" />
                        {!hasMultiplePhases ? <span>Duplicate phase</span> : null}
                      </button>
                      {hasMultiplePhases ? (
                        <button
                          type="button"
                          className="trainingRecentIconButton trainingRecentIconButton-danger"
                          onClick={() => {
                            const next = schedulers.filter(
                              (_, currentIndex) => currentIndex !== index
                            );
                            commitSchedulers(
                              next.length
                                ? next
                                : [defaultSchedulerForType("constant", maxSteps, baseLearningRate)]
                            );
                          }}
                          aria-label={`Remove phase ${index + 1}`}
                          title="Remove"
                        >
                          <FiTrash2 aria-hidden="true" />
                        </button>
                      ) : null}
                    </div>
                  </div>

                  <div className="lrPlannerPhaseRange">
                    Steps {formatInteger(phaseStart)} -{" "}
                    {formatInteger(Math.max(phaseStart, phaseEnd - 1))}
                  </div>

                  <div className="fieldGrid lrPlannerFieldGrid">
                    <label className="fieldLabel">
                      <span>Scheduler type</span>
                      <select
                        value={scheduler.type}
                        onChange={(event) => {
                          const nextType = event.target.value as LearningRateSchedulerType;
                          updateSchedulerAt(index, () =>
                            defaultSchedulerForType(nextType, scheduler.steps, baseLearningRate)
                          );
                        }}
                      >
                        {SCHEDULER_TYPES.map((type) => (
                          <option key={type} value={type}>
                            {SCHEDULER_DEFINITIONS[type].label}
                          </option>
                        ))}
                      </select>
                    </label>
                    <label className="fieldLabel">
                      <span>Phase steps</span>
                      <PlannerNumberInput
                        min={scheduler.type === "multistep" ? 2 : 1}
                        value={scheduler.steps}
                        onCommit={(value) =>
                          updateSchedulerAt(index, (current) => ({
                            ...current,
                            steps: Math.max(current.type === "multistep" ? 2 : 1, value),
                          }))
                        }
                      />
                    </label>
                    <SchedulerFields
                      scheduler={scheduler}
                      onPatch={(patch) =>
                        updateSchedulerAt(index, (current) => ({
                          ...current,
                          ...patch,
                        }))
                      }
                    />
                  </div>
                  <p className="fieldNote">{definition.description}</p>
                </article>
              );
            })}
          </div>

          <div className="lrPlannerChartPanel">
            <div className="lrPlannerSectionHead">
              <div>
                <strong>Predicted LR curve</strong>
                <span>
                  {formatInteger(projection.length)} chart point
                  {projection.length === 1 ? "" : "s"}
                </span>
              </div>
              <button
                type="button"
                className="buttonGhost buttonSmall"
                onClick={() =>
                  applySerializedSchedulers(makeWarmupCosineSchedule(maxSteps, baseLearningRate))
                }
              >
                <FiRefreshCw aria-hidden="true" /> Starter schedule
              </button>
            </div>

            <div className="lrPlannerChartFrame">
              {projection.length ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={projection} margin={LR_CHART_MARGIN}>
                    <CartesianGrid stroke="var(--line)" strokeDasharray="3 3" vertical={false} />
                    <XAxis
                      dataKey="step"
                      type="number"
                      domain={[0, xDomainEnd]}
                      tickLine={false}
                      axisLine={false}
                      minTickGap={24}
                      tick={{ fill: "var(--text-muted)", fontSize: 11 }}
                      tickFormatter={(value) => String(value)}
                    />
                    <YAxis
                      width={LR_CHART_Y_AXIS_WIDTH}
                      domain={yDomain}
                      tickLine={false}
                      axisLine={false}
                      tick={{ fill: "var(--text-muted)", fontSize: 11 }}
                      tickFormatter={(value) => formatLearningRate(Number(value))}
                    />
                    <Tooltip
                      cursor={{ stroke: "var(--text-muted)", strokeDasharray: "4 4" }}
                      content={<LearningRateTooltip />}
                    />
                    <Line
                      type="linear"
                      dataKey="lr"
                      stroke="var(--brand)"
                      strokeWidth={2.4}
                      dot={false}
                      activeDot={{ r: 4, strokeWidth: 2, stroke: "var(--brand)" }}
                      isAnimationActive={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="trainingEmpty">
                  Set a positive base learning rate to preview the curve.
                </div>
              )}
            </div>

            <div
              className="lrPlannerTimeline"
              style={timelineStyle}
              aria-label="Learning-rate phase timeline"
            >
              <span className="lrPlannerTimelineGutter" aria-hidden="true" />
              <div className="lrPlannerTimelineTrack">
                {schedulers.map((scheduler, index) => {
                  const width =
                    totalPlannedSteps > 0 ? (scheduler.steps / totalPlannedSteps) * 100 : 0;
                  return (
                    <span
                      key={`timeline-${index}-${scheduler.type}`}
                      className="lrPlannerTimelineSegment"
                      style={{ width: `${width}%` }}
                      title={`Phase ${index + 1}: ${SCHEDULER_DEFINITIONS[scheduler.type].label}`}
                    >
                      {SCHEDULER_DEFINITIONS[scheduler.type].shortLabel}
                    </span>
                  );
                })}
              </div>
              <span className="lrPlannerTimelineGutter" aria-hidden="true" />
            </div>

            <div className="lrPlannerInsightGrid">
              <div>
                <span>First logged LR</span>
                <strong>{formatLearningRate(stats?.first)}</strong>
              </div>
              <div>
                <span>Lowest LR</span>
                <strong>{formatLearningRate(stats?.min)}</strong>
              </div>
              <div>
                <span>Backend payload</span>
                <strong>sequential</strong>
              </div>
            </div>

            {!exactStepFit ? (
              <p className="lrPlannerWarning">
                The backend requires scheduler phase steps to equal maximum training steps.
              </p>
            ) : (
              <p className="fieldNote">
                The projection uses the current optimizer learning rate and the exact scheduler
                sequence sent to the backend.
              </p>
            )}
          </div>
        </div>
      </div>
    </details>
  );
}
