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

type ConfigNumberMode = "integer" | "decimal" | "scientific";

export type LearningRateSchedulerType =
  | "linear"
  | "cosine_annealing"
  | "step"
  | "multistep"
  | "exponential"
  | "constant"
  | "cosine_annealing_warm_restarts";

export type LearningRateSchedulerConfig = Record<string, unknown> & {
  type: LearningRateSchedulerType;
  steps: number;
};

interface LearningRateSchedulePlannerProps {
  baseLearningRate: number;
  maxSteps: number;
  schedulerConfig: Record<string, unknown>;
  onSchedulersChange: (schedulers: Record<string, unknown>[]) => void;
}

interface SchedulerDefinition {
  label: string;
  shortLabel: string;
  description: string;
  bestFor: string;
}

interface ProjectionPoint {
  step: number;
  lr: number;
  phaseIndex: number;
  phaseLabel: string;
}

const SCHEDULER_TYPES: LearningRateSchedulerType[] = [
  "linear",
  "cosine_annealing",
  "step",
  "multistep",
  "exponential",
  "constant",
  "cosine_annealing_warm_restarts",
];

const SCHEDULER_DEFINITIONS: Record<LearningRateSchedulerType, SchedulerDefinition> = {
  linear: {
    label: "Linear",
    shortLabel: "Linear",
    description: "Move smoothly between two LR factors over a fixed phase.",
    bestFor: "Warmup, cooldown, and deliberate ramps.",
  },
  cosine_annealing: {
    label: "Cosine annealing",
    shortLabel: "Cosine",
    description: "Decay from the base LR to an absolute minimum with a cosine curve.",
    bestFor: "Default fine-tuning or pretraining tails.",
  },
  step: {
    label: "Step decay",
    shortLabel: "Step",
    description: "Multiply the LR by gamma every fixed number of steps.",
    bestFor: "Coarse staged drops.",
  },
  multistep: {
    label: "Milestone decay",
    shortLabel: "Milestone",
    description: "Multiply the LR at explicit milestone steps inside the phase.",
    bestFor: "Hand-authored decay plans.",
  },
  exponential: {
    label: "Exponential",
    shortLabel: "Expo",
    description: "Multiply the LR by gamma every step.",
    bestFor: "Smooth multiplicative decay.",
  },
  constant: {
    label: "Constant factor",
    shortLabel: "Constant",
    description: "Hold a fixed multiplicative factor for the whole phase.",
    bestFor: "Plateaus and controlled holds.",
  },
  cosine_annealing_warm_restarts: {
    label: "Cosine warm restarts",
    shortLabel: "Restarts",
    description: "Run repeated cosine cycles with optional cycle growth.",
    bestFor: "Exploration bursts or restart-heavy schedules.",
  },
};

const MAX_CHART_POINTS = 900;
const LR_CHART_MARGIN = { top: 12, right: 12, bottom: 0, left: 6 };
const LR_CHART_Y_AXIS_WIDTH = 62;
const LR_CHART_FRAME_INSET = 9;

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function asRecord(value: unknown): Record<string, unknown> {
  return isRecord(value) ? value : {};
}

function asNumber(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

function asPositiveNumber(value: unknown, fallback: number, allowZero = false): number {
  const parsed = typeof value === "number" ? value : typeof value === "string" ? Number(value) : NaN;
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  if (allowZero) {
    return Math.max(0, parsed);
  }
  return parsed > 0 ? parsed : fallback;
}

function asPositiveInteger(value: unknown, fallback: number): number {
  const parsed = typeof value === "number" ? value : typeof value === "string" ? Number(value) : NaN;
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.max(1, Math.trunc(parsed));
}

function isSchedulerType(value: unknown): value is LearningRateSchedulerType {
  return SCHEDULER_TYPES.includes(value as LearningRateSchedulerType);
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function formatInteger(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "n/a";
  }
  return Math.trunc(value).toLocaleString();
}

function formatExponentialValue(value: number, digits = 3): string {
  if (!Number.isFinite(value)) {
    return "n/a";
  }
  const [mantissa = "", exponent = "0"] = value.toExponential(digits).split("e");
  const trimmedMantissa = mantissa.replace(/\.?0+$/, "");
  const exponentValue = Number(exponent);
  const exponentSign = exponentValue >= 0 ? "+" : "";
  return `${trimmedMantissa}e${exponentSign}${exponentValue}`;
}

function formatLearningRate(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "n/a";
  }
  return formatExponentialValue(value, 3);
}

function formatNumberInputValue(value: number, mode: ConfigNumberMode = "integer"): string {
  if (!Number.isFinite(value)) {
    return "";
  }
  if (mode === "scientific") {
    return formatExponentialValue(value, 3);
  }
  const asText = String(value);
  if (!/[eE]/.test(asText)) {
    return asText;
  }
  return value.toLocaleString("en-US", {
    useGrouping: false,
    maximumFractionDigits: 20,
  });
}

function sanitizePositiveDecimalInput(value: string): string {
  const digitsAndDot = value.replace(/[^0-9.]/g, "");
  const firstDotIndex = digitsAndDot.indexOf(".");
  if (firstDotIndex === -1) {
    return digitsAndDot;
  }
  return `${digitsAndDot.slice(0, firstDotIndex + 1)}${digitsAndDot
    .slice(firstDotIndex + 1)
    .replace(/\./g, "")}`;
}

function sanitizePositiveScientificInput(value: string): string {
  const compact = value.replace(/,/g, "").replace(/\s/g, "").toLowerCase();
  let output = "";
  let hasDot = false;
  let hasExponent = false;
  let canAddExponentSign = false;

  for (const char of compact) {
    if (/[0-9]/.test(char)) {
      output += char;
      canAddExponentSign = false;
      continue;
    }

    if (char === "." && !hasDot && !hasExponent) {
      output += char;
      hasDot = true;
      canAddExponentSign = false;
      continue;
    }

    if (char === "e" && !hasExponent && output !== "" && output !== ".") {
      output += char;
      hasExponent = true;
      canAddExponentSign = true;
      continue;
    }

    if ((char === "-" || char === "+") && canAddExponentSign) {
      output += char;
      canAddExponentSign = false;
    }
  }

  return output;
}

function sanitizePositiveIntegerInput(value: string): string {
  return value.replace(/[^0-9]/g, "");
}

function parseConfigNumberInput(value: string, mode: ConfigNumberMode): number | null {
  const trimmed = value.trim();
  if (trimmed === "" || trimmed === "." || /e[+-]?$/i.test(trimmed)) {
    return null;
  }
  const parsed = Number(trimmed);
  if (!Number.isFinite(parsed)) {
    return null;
  }
  if (mode === "integer" && !Number.isInteger(parsed)) {
    return null;
  }
  return parsed;
}

function parseMilestonesInput(value: string, steps: number): number[] {
  const milestones = value
    .split(/[\s,]+/)
    .map((item) => Number(item.trim()))
    .filter((item) => Number.isInteger(item) && item > 0 && item < steps)
    .map((item) => Math.trunc(item));
  return Array.from(new Set(milestones)).sort((left, right) => left - right);
}

function defaultMilestones(steps: number): number[] {
  if (steps <= 1) {
    return [];
  }
  const first = Math.max(1, Math.min(steps - 1, Math.round(steps * 0.5)));
  const second = Math.max(1, Math.min(steps - 1, Math.round(steps * 0.75)));
  return Array.from(new Set([first, second])).sort((left, right) => left - right);
}

function defaultSchedulerForType(
  type: LearningRateSchedulerType,
  steps: number,
  baseLearningRate: number
): LearningRateSchedulerConfig {
  const safeSteps = Math.max(type === "multistep" ? 2 : 1, Math.trunc(steps));
  const etaMin = Math.max(0, baseLearningRate * 0.03);

  if (type === "linear") {
    return {
      type,
      steps: safeSteps,
      start_factor: 0.1,
      end_factor: 1,
    };
  }
  if (type === "cosine_annealing") {
    return {
      type,
      steps: safeSteps,
      eta_min: etaMin,
    };
  }
  if (type === "step") {
    return {
      type,
      steps: safeSteps,
      step_size: Math.max(1, Math.round(safeSteps / 3)),
      gamma: 0.5,
    };
  }
  if (type === "multistep") {
    return {
      type,
      steps: safeSteps,
      milestones: defaultMilestones(safeSteps),
      gamma: 0.5,
    };
  }
  if (type === "exponential") {
    return {
      type,
      steps: safeSteps,
      gamma: 0.99,
    };
  }
  if (type === "cosine_annealing_warm_restarts") {
    return {
      type,
      steps: safeSteps,
      t_0: Math.max(1, Math.round(safeSteps / 4)),
      t_mult: 2,
      eta_min: etaMin,
    };
  }
  return {
    type: "constant",
    steps: safeSteps,
    factor: 1,
  };
}

function sanitizeSchedulerRecord(
  rawScheduler: unknown,
  fallbackSteps: number,
  baseLearningRate: number
): LearningRateSchedulerConfig {
  const record = asRecord(rawScheduler);
  const requestedType = isSchedulerType(record.type) ? record.type : "constant";
  const requestedSteps = asPositiveInteger(record.steps, fallbackSteps);
  const type =
    requestedType === "multistep" && requestedSteps <= 1 ? "constant" : requestedType;
  const defaults = defaultSchedulerForType(type, requestedSteps, baseLearningRate);
  const steps = Math.max(type === "multistep" ? 2 : 1, requestedSteps);

  if (type === "linear") {
    return {
      type,
      steps,
      start_factor: asPositiveNumber(record.start_factor, asNumber(defaults.start_factor, 0.1)),
      end_factor: asPositiveNumber(record.end_factor, asNumber(defaults.end_factor, 1)),
    };
  }
  if (type === "cosine_annealing") {
    return {
      type,
      steps,
      eta_min: asPositiveNumber(record.eta_min, asNumber(defaults.eta_min, 0), true),
    };
  }
  if (type === "step") {
    return {
      type,
      steps,
      step_size: clamp(
        asPositiveInteger(record.step_size, asNumber(defaults.step_size, 1)),
        1,
        steps
      ),
      gamma: asPositiveNumber(record.gamma, asNumber(defaults.gamma, 0.5)),
    };
  }
  if (type === "multistep") {
    const currentMilestones = Array.isArray(record.milestones)
      ? record.milestones.join(",")
      : String(record.milestones ?? "");
    const milestones = parseMilestonesInput(currentMilestones, steps);
    return {
      type,
      steps,
      milestones: milestones.length ? milestones : defaultMilestones(steps),
      gamma: asPositiveNumber(record.gamma, asNumber(defaults.gamma, 0.5)),
    };
  }
  if (type === "exponential") {
    return {
      type,
      steps,
      gamma: asPositiveNumber(record.gamma, asNumber(defaults.gamma, 0.99)),
    };
  }
  if (type === "cosine_annealing_warm_restarts") {
    return {
      type,
      steps,
      t_0: asPositiveInteger(record.t_0, asNumber(defaults.t_0, 1)),
      t_mult: Math.max(1, asPositiveInteger(record.t_mult, asNumber(defaults.t_mult, 1))),
      eta_min: asPositiveNumber(record.eta_min, asNumber(defaults.eta_min, 0), true),
    };
  }
  return {
    type: "constant",
    steps,
    factor: asPositiveNumber(record.factor, asNumber(defaults.factor, 1)),
  };
}

function normalizeSchedulers(
  rawSchedulers: unknown,
  maxSteps: number,
  baseLearningRate: number
): LearningRateSchedulerConfig[] {
  const targetSteps = Math.max(1, Math.trunc(maxSteps || 1));
  if (!Array.isArray(rawSchedulers) || rawSchedulers.length === 0) {
    return [defaultSchedulerForType("constant", targetSteps, baseLearningRate)];
  }

  return rawSchedulers.map((scheduler, index) =>
    sanitizeSchedulerRecord(
      scheduler,
      index === rawSchedulers.length - 1
        ? targetSteps
        : Math.max(1, Math.round(targetSteps / rawSchedulers.length)),
      baseLearningRate
    )
  );
}

function serializeScheduler(scheduler: LearningRateSchedulerConfig): Record<string, unknown> {
  if (scheduler.type === "linear") {
    return {
      type: scheduler.type,
      steps: scheduler.steps,
      start_factor: scheduler.start_factor,
      end_factor: scheduler.end_factor,
    };
  }
  if (scheduler.type === "cosine_annealing") {
    return {
      type: scheduler.type,
      steps: scheduler.steps,
      eta_min: scheduler.eta_min,
    };
  }
  if (scheduler.type === "step") {
    return {
      type: scheduler.type,
      steps: scheduler.steps,
      step_size: scheduler.step_size,
      gamma: scheduler.gamma,
    };
  }
  if (scheduler.type === "multistep") {
    return {
      type: scheduler.type,
      steps: scheduler.steps,
      milestones: scheduler.milestones,
      gamma: scheduler.gamma,
    };
  }
  if (scheduler.type === "exponential") {
    return {
      type: scheduler.type,
      steps: scheduler.steps,
      gamma: scheduler.gamma,
    };
  }
  if (scheduler.type === "cosine_annealing_warm_restarts") {
    return {
      type: scheduler.type,
      steps: scheduler.steps,
      t_0: scheduler.t_0,
      t_mult: scheduler.t_mult,
      eta_min: scheduler.eta_min,
    };
  }
  return {
    type: "constant",
    steps: scheduler.steps,
    factor: scheduler.factor,
  };
}

function sanitizeSchedulerAfterStepChange(
  scheduler: Record<string, unknown>,
  baseLearningRate: number
): Record<string, unknown> {
  return serializeScheduler(
    sanitizeSchedulerRecord(scheduler, asPositiveInteger(scheduler.steps, 1), baseLearningRate)
  );
}

export function fitSchedulersToMaxSteps(
  rawSchedulers: Record<string, unknown>[],
  maxSteps: number,
  baseLearningRate = 0.0003
): Record<string, unknown>[] {
  const targetSteps = Math.max(1, Math.trunc(maxSteps || 1));
  const sourceSchedulers =
    rawSchedulers.length > 0
      ? rawSchedulers
      : [defaultSchedulerForType("constant", targetSteps, baseLearningRate)];
  const usableSchedulers = sourceSchedulers.slice(0, targetSteps);
  const sourceTotal = usableSchedulers.reduce(
    (sum, scheduler) => sum + asPositiveInteger(scheduler.steps, 1),
    0
  );

  if (usableSchedulers.length === 1 || sourceTotal <= 0) {
    return [
      sanitizeSchedulerAfterStepChange(
        {
          ...usableSchedulers[0],
          steps: targetSteps,
        },
        baseLearningRate
      ),
    ];
  }

  const provisional = usableSchedulers.map((scheduler) => {
    const rawSteps = asPositiveInteger(scheduler.steps, 1);
    const exact = (rawSteps / sourceTotal) * targetSteps;
    return {
      scheduler,
      exact,
      steps: Math.max(1, Math.floor(exact)),
      remainder: exact - Math.floor(exact),
    };
  });

  let allocated = provisional.reduce((sum, item) => sum + item.steps, 0);
  while (allocated > targetSteps) {
    const candidate = provisional
      .map((item, index) => ({ ...item, index }))
      .filter((item) => item.steps > 1)
      .sort((left, right) => left.remainder - right.remainder)[0];
    if (!candidate) {
      break;
    }
    provisional[candidate.index].steps -= 1;
    allocated -= 1;
  }
  while (allocated < targetSteps) {
    const candidate = provisional
      .map((item, index) => ({ ...item, index }))
      .sort((left, right) => right.remainder - left.remainder)[0];
    provisional[candidate.index].steps += 1;
    allocated += 1;
  }

  return provisional.map((item) =>
    sanitizeSchedulerAfterStepChange(
      {
        ...item.scheduler,
        steps: item.steps,
      },
      baseLearningRate
    )
  );
}

function makeWarmupCosineSchedule(
  maxSteps: number,
  baseLearningRate: number
): Record<string, unknown>[] {
  const targetSteps = Math.max(1, Math.trunc(maxSteps || 1));
  if (targetSteps === 1) {
    return [serializeScheduler(defaultSchedulerForType("constant", 1, baseLearningRate))];
  }
  const warmupSteps = clamp(Math.round(targetSteps * 0.08), 1, Math.max(1, targetSteps - 1));
  return [
    {
      type: "linear",
      steps: warmupSteps,
      start_factor: 0.1,
      end_factor: 1,
    },
    {
      type: "cosine_annealing",
      steps: targetSteps - warmupSteps,
      eta_min: Math.max(0, baseLearningRate * 0.03),
    },
  ];
}

function makeWarmupHoldSchedule(maxSteps: number): Record<string, unknown>[] {
  const targetSteps = Math.max(1, Math.trunc(maxSteps || 1));
  if (targetSteps === 1) {
    return [{ type: "constant", steps: 1, factor: 1 }];
  }
  const warmupSteps = clamp(Math.round(targetSteps * 0.1), 1, Math.max(1, targetSteps - 1));
  return [
    {
      type: "linear",
      steps: warmupSteps,
      start_factor: 0.1,
      end_factor: 1,
    },
    {
      type: "constant",
      steps: targetSteps - warmupSteps,
      factor: 1,
    },
  ];
}

function makeStepDecaySchedule(maxSteps: number): Record<string, unknown>[] {
  const targetSteps = Math.max(1, Math.trunc(maxSteps || 1));
  return [
    {
      type: "step",
      steps: targetSteps,
      step_size: Math.max(1, Math.round(targetSteps / 3)),
      gamma: 0.5,
    },
  ];
}

function sumSchedulerSteps(schedulers: Array<{ steps: number }>): number {
  return schedulers.reduce((sum, scheduler) => sum + scheduler.steps, 0);
}

function localLrForScheduler(
  scheduler: LearningRateSchedulerConfig,
  localStep: number,
  baseLearningRate: number
): number {
  const steps = Math.max(1, scheduler.steps);
  const step = Math.max(0, localStep);

  if (scheduler.type === "linear") {
    const startFactor = asPositiveNumber(scheduler.start_factor, 0.1);
    const endFactor = asPositiveNumber(scheduler.end_factor, 1);
    const progress = clamp(step / steps, 0, 1);
    return baseLearningRate * (startFactor + (endFactor - startFactor) * progress);
  }

  if (scheduler.type === "cosine_annealing") {
    const etaMin = asPositiveNumber(scheduler.eta_min, 0, true);
    const progress = clamp(step / steps, 0, 1);
    return etaMin + ((baseLearningRate - etaMin) * (1 + Math.cos(Math.PI * progress))) / 2;
  }

  if (scheduler.type === "step") {
    const stepSize = Math.max(1, asPositiveInteger(scheduler.step_size, 1));
    const gamma = asPositiveNumber(scheduler.gamma, 0.1);
    return baseLearningRate * gamma ** Math.floor(step / stepSize);
  }

  if (scheduler.type === "multistep") {
    const milestones = Array.isArray(scheduler.milestones)
      ? scheduler.milestones.filter((item): item is number => typeof item === "number")
      : [];
    const gamma = asPositiveNumber(scheduler.gamma, 0.1);
    const decayCount = milestones.filter((milestone) => milestone <= step).length;
    return baseLearningRate * gamma ** decayCount;
  }

  if (scheduler.type === "exponential") {
    const gamma = asPositiveNumber(scheduler.gamma, 0.99);
    return baseLearningRate * gamma ** step;
  }

  if (scheduler.type === "cosine_annealing_warm_restarts") {
    const etaMin = asPositiveNumber(scheduler.eta_min, 0, true);
    let cycleLength = Math.max(1, asPositiveInteger(scheduler.t_0, 1));
    const cycleMultiplier = Math.max(1, asPositiveInteger(scheduler.t_mult, 1));
    let cycleStep = step;
    while (cycleStep >= cycleLength) {
      cycleStep -= cycleLength;
      cycleLength *= cycleMultiplier;
    }
    const progress = cycleStep / cycleLength;
    return etaMin + ((baseLearningRate - etaMin) * (1 + Math.cos(Math.PI * progress))) / 2;
  }

  return baseLearningRate * asPositiveNumber(scheduler.factor, 1);
}

function buildLearningRateProjection(
  schedulers: LearningRateSchedulerConfig[],
  baseLearningRate: number
): ProjectionPoint[] {
  const totalSteps = sumSchedulerSteps(schedulers);
  if (totalSteps <= 0 || !Number.isFinite(baseLearningRate) || baseLearningRate <= 0) {
    return [];
  }

  const stride = Math.max(1, Math.floor(totalSteps / MAX_CHART_POINTS));
  const points: ProjectionPoint[] = [];
  let globalStep = 0;

  schedulers.forEach((scheduler, phaseIndex) => {
    const phaseLabel = SCHEDULER_DEFINITIONS[scheduler.type].shortLabel;
    for (let localStep = 0; localStep < scheduler.steps; localStep += 1) {
      const isBoundary = localStep === 0 || localStep === scheduler.steps - 1;
      if (globalStep % stride === 0 || isBoundary) {
        points.push({
          step: globalStep,
          lr: localLrForScheduler(scheduler, localStep, baseLearningRate),
          phaseIndex,
          phaseLabel,
        });
      }
      globalStep += 1;
    }
  });

  const lastScheduler = schedulers[schedulers.length - 1];
  if (lastScheduler) {
    points.push({
      step: totalSteps,
      lr: localLrForScheduler(lastScheduler, lastScheduler.steps, baseLearningRate),
      phaseIndex: schedulers.length - 1,
      phaseLabel: SCHEDULER_DEFINITIONS[lastScheduler.type].shortLabel,
    });
  }

  const deduped = new Map<string, ProjectionPoint>();
  points.forEach((point) => {
    deduped.set(`${point.step}-${point.phaseIndex}-${point.lr}`, point);
  });
  return Array.from(deduped.values()).sort((left, right) => left.step - right.step);
}

function projectionStats(points: ProjectionPoint[]) {
  if (!points.length) {
    return null;
  }
  let min = points[0].lr;
  let max = points[0].lr;
  points.forEach((point) => {
    min = Math.min(min, point.lr);
    max = Math.max(max, point.lr);
  });
  return {
    first: points[0].lr,
    final: points[points.length - 1].lr,
    min,
    max,
  };
}

function lrAxisDomain(points: ProjectionPoint[]): [number, number] {
  if (!points.length) {
    return [0, 1];
  }
  const values = points.map((point) => point.lr);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const spread = max - min;
  const padding = spread === 0 ? Math.max(Math.abs(max) * 0.08, 1e-8) : spread * 0.08;
  return [Math.max(0, min - padding), max + padding];
}

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
      onBlur={(event) => {
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
