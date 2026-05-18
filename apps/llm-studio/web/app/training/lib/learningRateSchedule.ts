export type ConfigNumberMode = "integer" | "decimal" | "scientific";

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

export interface LearningRateSchedulePlannerProps {
  baseLearningRate: number;
  maxSteps: number;
  schedulerConfig: Record<string, unknown>;
  onSchedulersChange: (schedulers: Record<string, unknown>[]) => void;
}

export interface SchedulerDefinition {
  label: string;
  shortLabel: string;
  description: string;
  bestFor: string;
}

export interface ProjectionPoint {
  step: number;
  lr: number;
  phaseIndex: number;
  phaseLabel: string;
}

export const SCHEDULER_TYPES: LearningRateSchedulerType[] = [
  "linear",
  "cosine_annealing",
  "step",
  "multistep",
  "exponential",
  "constant",
  "cosine_annealing_warm_restarts",
];

export const SCHEDULER_DEFINITIONS: Record<LearningRateSchedulerType, SchedulerDefinition> = {
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

export const MAX_CHART_POINTS = 900;
export const LR_CHART_MARGIN = { top: 12, right: 12, bottom: 0, left: 6 };
export const LR_CHART_Y_AXIS_WIDTH = 62;
export const LR_CHART_FRAME_INSET = 9;

export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function asRecord(value: unknown): Record<string, unknown> {
  return isRecord(value) ? value : {};
}

export function asNumber(value: unknown, fallback: number): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

export function asPositiveNumber(value: unknown, fallback: number, allowZero = false): number {
  const parsed = typeof value === "number" ? value : typeof value === "string" ? Number(value) : NaN;
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  if (allowZero) {
    return Math.max(0, parsed);
  }
  return parsed > 0 ? parsed : fallback;
}

export function asPositiveInteger(value: unknown, fallback: number): number {
  const parsed = typeof value === "number" ? value : typeof value === "string" ? Number(value) : NaN;
  if (!Number.isFinite(parsed)) {
    return fallback;
  }
  return Math.max(1, Math.trunc(parsed));
}

export function isSchedulerType(value: unknown): value is LearningRateSchedulerType {
  return SCHEDULER_TYPES.includes(value as LearningRateSchedulerType);
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function formatInteger(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "N/A";
  }
  return Math.trunc(value).toLocaleString();
}

export function formatExponentialValue(value: number, digits = 3): string {
  if (!Number.isFinite(value)) {
    return "N/A";
  }
  const [mantissa = "", exponent = "0"] = value.toExponential(digits).split("e");
  const trimmedMantissa = mantissa.replace(/\.?0+$/, "");
  const exponentValue = Number(exponent);
  const exponentSign = exponentValue >= 0 ? "+" : "";
  return `${trimmedMantissa}e${exponentSign}${exponentValue}`;
}

export function formatLearningRate(value: number | null | undefined): string {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "N/A";
  }
  return formatExponentialValue(value, 3);
}

export function formatNumberInputValue(value: number, mode: ConfigNumberMode = "integer"): string {
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

export function sanitizePositiveDecimalInput(value: string): string {
  const digitsAndDot = value.replace(/[^0-9.]/g, "");
  const firstDotIndex = digitsAndDot.indexOf(".");
  if (firstDotIndex === -1) {
    return digitsAndDot;
  }
  return `${digitsAndDot.slice(0, firstDotIndex + 1)}${digitsAndDot
    .slice(firstDotIndex + 1)
    .replace(/\./g, "")}`;
}

export function sanitizePositiveScientificInput(value: string): string {
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

export function sanitizePositiveIntegerInput(value: string): string {
  return value.replace(/[^0-9]/g, "");
}

export function parseConfigNumberInput(value: string, mode: ConfigNumberMode): number | null {
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

export function parseMilestonesInput(value: string, steps: number): number[] {
  const milestones = value
    .split(/[\s,]+/)
    .map((item) => Number(item.trim()))
    .filter((item) => Number.isInteger(item) && item > 0 && item < steps)
    .map((item) => Math.trunc(item));
  return Array.from(new Set(milestones)).sort((left, right) => left - right);
}

export function defaultMilestones(steps: number): number[] {
  if (steps <= 1) {
    return [];
  }
  const first = Math.max(1, Math.min(steps - 1, Math.round(steps * 0.5)));
  const second = Math.max(1, Math.min(steps - 1, Math.round(steps * 0.75)));
  return Array.from(new Set([first, second])).sort((left, right) => left - right);
}

export function defaultSchedulerForType(
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

export function sanitizeSchedulerRecord(
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

export function normalizeSchedulers(
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

export function serializeScheduler(scheduler: LearningRateSchedulerConfig): Record<string, unknown> {
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

export function sanitizeSchedulerAfterStepChange(
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

export function makeWarmupCosineSchedule(
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

export function makeWarmupHoldSchedule(maxSteps: number): Record<string, unknown>[] {
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

export function makeStepDecaySchedule(maxSteps: number): Record<string, unknown>[] {
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

export function sumSchedulerSteps(schedulers: Array<{ steps: number }>): number {
  return schedulers.reduce((sum, scheduler) => sum + scheduler.steps, 0);
}

export function localLrForScheduler(
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

export function buildLearningRateProjection(
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

export function projectionStats(points: ProjectionPoint[]) {
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

export function lrAxisDomain(points: ProjectionPoint[]): [number, number] {
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
