import { ACTIVATION_TYPES, type BlockComponent } from "../../../lib/defaults";
import type {
  StudioComponentKind,
  StudioComponentPrefab,
} from "../types";
import { labelForComponentKind } from "../utils/document";

export const MAX_DOCUMENT_HISTORY_ENTRIES = 100;

export function pruneIdSet(current: Set<string>, validIds: Set<string>): Set<string> {
  let changed = false;
  const next = new Set<string>();

  current.forEach((id) => {
    if (validIds.has(id)) {
      next.add(id);
      return;
    }
    changed = true;
  });

  return changed ? next : current;
}

function isStudioComponentKind(value: unknown): value is StudioComponentKind {
  return value === "attention" || value === "mlp" || value === "norm" || value === "activation";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function isValidActivationType(value: unknown): boolean {
  return typeof value === "string" && (ACTIVATION_TYPES as readonly string[]).includes(value);
}

function isValidNormConfig(value: unknown): boolean {
  if (!isRecord(value) || typeof value.type !== "string") {
    return false;
  }
  if (value.type === "layernorm") {
    return true;
  }
  if (value.type !== "rmsnorm") {
    return false;
  }
  return typeof value.learnable_gamma === "boolean";
}

function isValidMlpStepConfig(value: unknown): boolean {
  if (!isRecord(value)) {
    return false;
  }
  if ("linear" in value) {
    return isRecord(value.linear) && typeof value.linear.bias === "boolean";
  }
  if ("norm" in value) {
    return isValidNormConfig(value.norm);
  }
  if ("activation" in value) {
    return isRecord(value.activation) && isValidActivationType(value.activation.type);
  }
  return false;
}

function isValidComponentConfig(kind: StudioComponentKind, value: unknown): value is BlockComponent {
  if (!isRecord(value)) {
    return false;
  }

  if (kind === "attention") {
    if (!("attention" in value) || !isRecord(value.attention)) {
      return false;
    }
    return (
      isFiniteNumber(value.attention.n_head) &&
      isFiniteNumber(value.attention.n_kv_head)
    );
  }

  if (kind === "mlp") {
    if (!("mlp" in value) || !isRecord(value.mlp)) {
      return false;
    }
    if (!isFiniteNumber(value.mlp.multiplier) || !Array.isArray(value.mlp.sequence)) {
      return false;
    }
    return value.mlp.sequence.every(isValidMlpStepConfig);
  }

  if (kind === "norm") {
    return "norm" in value && isValidNormConfig(value.norm);
  }

  return (
    "activation" in value &&
    isRecord(value.activation) &&
    isValidActivationType(value.activation.type)
  );
}

export function parseStoredComponentPrefabs(raw: string | null): StudioComponentPrefab[] {
  if (!raw) {
    return [];
  }

  try {
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) {
      return [];
    }

    return parsed.flatMap((item) => {
      if (!item || typeof item !== "object") {
        return [];
      }
      const candidate = item as {
        id?: unknown;
        name?: unknown;
        kind?: unknown;
        component?: unknown;
        createdAt?: unknown;
      };

      if (typeof candidate.id !== "string" || candidate.id.trim() === "") {
        return [];
      }
      if (typeof candidate.name !== "string" || candidate.name.trim() === "") {
        return [];
      }
      if (!isStudioComponentKind(candidate.kind)) {
        return [];
      }
      if (!isValidComponentConfig(candidate.kind, candidate.component)) {
        return [];
      }

      return [
        {
          id: candidate.id,
          name: candidate.name,
          kind: candidate.kind,
          component: candidate.component,
          createdAt: typeof candidate.createdAt === "number" ? candidate.createdAt : Date.now(),
        } satisfies StudioComponentPrefab,
      ];
    });
  } catch {
    return [];
  }
}

export function createUniquePrefabName(
  kind: StudioComponentKind,
  currentPrefabs: StudioComponentPrefab[]
): string {
  const base = `${labelForComponentKind(kind)} prefab`;
  const existingNames = new Set(currentPrefabs.map((prefab) => prefab.name.toLowerCase()));
  let index = 1;
  while (existingNames.has(`${base} ${index}`.toLowerCase())) {
    index += 1;
  }
  return `${base} ${index}`;
}

export function createUniquePrefabNameFromBase(
  baseName: string,
  currentPrefabs: StudioComponentPrefab[],
  excludePrefabId?: string
): string {
  const normalizedBase = baseName.trim();
  if (!normalizedBase) {
    return "Prefab";
  }

  const existingNames = new Set(
    currentPrefabs
      .filter((prefab) => prefab.id !== excludePrefabId)
      .map((prefab) => prefab.name.toLowerCase())
  );
  if (!existingNames.has(normalizedBase.toLowerCase())) {
    return normalizedBase;
  }

  let suffix = 2;
  while (existingNames.has(`${normalizedBase} (${suffix})`.toLowerCase())) {
    suffix += 1;
  }
  return `${normalizedBase} (${suffix})`;
}
