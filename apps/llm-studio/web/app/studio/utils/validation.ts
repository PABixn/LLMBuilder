import {
  ACTIVATION_TYPES,
  type ActivationComponentConfig,
  type ActivationType,
  type BlockComponent,
  type LayerNormConfig,
  type MlpStep,
  type ModelBlock,
  type ModelConfig,
  type NormComponentConfig,
  type NormConfig,
  type RMSNormConfig,
} from "../../../lib/defaults";

import { createId } from "./document";
import type {
  BaseIntegerModelField,
  Diagnostic,
  DiagnosticLevel,
  DiagnosticSource,
  ImportedModelConfigParseResult,
} from "../types";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function pushDiagnostic(
  diagnostics: Diagnostic[],
  level: DiagnosticLevel,
  source: DiagnosticSource,
  path: string,
  message: string
): void {
  diagnostics.push({
    id: createId("diag"),
    level,
    source,
    path,
    message,
  });
}

export function validateLocalConfig(config: ModelConfig): Diagnostic[] {
  const diagnostics: Diagnostic[] = [];

  const integerFields: BaseIntegerModelField[] = ["context_length", "vocab_size", "n_embd"];

  for (const field of integerFields) {
    const value = config[field];
    if (!Number.isFinite(value) || !Number.isInteger(value) || value < 1) {
      pushDiagnostic(diagnostics, "error", "local", field, "Must be an integer greater than 0.");
    }
  }

  if (!Array.isArray(config.blocks) || config.blocks.length === 0) {
    pushDiagnostic(diagnostics, "error", "local", "blocks", "At least one block is required.");
    return diagnostics;
  }

  config.blocks.forEach((block, blockIndex) => {
    const blockPath = `blocks[${blockIndex}]`;
    if (!Array.isArray(block.components) || block.components.length === 0) {
      pushDiagnostic(
        diagnostics,
        "error",
        "local",
        `${blockPath}.components`,
        "Block must contain at least one component."
      );
      return;
    }

    const hasAttention = block.components.some((component) => "attention" in component);
    const hasMlp = block.components.some((component) => "mlp" in component);
    if (!hasAttention) {
      pushDiagnostic(
        diagnostics,
        "warning",
        "local",
        blockPath,
        "Block has no attention component. This is allowed but unusual for transformer blocks."
      );
    }
    if (!hasMlp) {
      pushDiagnostic(
        diagnostics,
        "warning",
        "local",
        blockPath,
        "Block has no MLP component. Residual capacity may be limited."
      );
    }

    block.components.forEach((component, componentIndex) => {
      const componentPath = `${blockPath}.components[${componentIndex}]`;

      if ("attention" in component) {
        const { n_head, n_kv_head } = component.attention;
        if (!Number.isInteger(n_head) || n_head < 1) {
          pushDiagnostic(
            diagnostics,
            "error",
            "local",
            `${componentPath}.attention.n_head`,
            "n_head must be an integer greater than 0."
          );
        }
        if (!Number.isInteger(n_kv_head) || n_kv_head < 1) {
          pushDiagnostic(
            diagnostics,
            "error",
            "local",
            `${componentPath}.attention.n_kv_head`,
            "n_kv_head must be an integer greater than 0."
          );
        }
        if (Number.isInteger(n_head) && Number.isInteger(n_kv_head) && n_head > 0 && n_kv_head > 0) {
          if (n_kv_head > n_head) {
            pushDiagnostic(
              diagnostics,
              "error",
              "local",
              `${componentPath}.attention.n_kv_head`,
              "n_kv_head cannot exceed n_head."
            );
          }
          if (n_head % n_kv_head !== 0) {
            pushDiagnostic(
              diagnostics,
              "error",
              "local",
              `${componentPath}.attention`,
              "n_head must be divisible by n_kv_head (GQA grouping constraint)."
            );
          }
          if (Number.isInteger(config.n_embd) && config.n_embd > 0 && config.n_embd % n_head !== 0) {
            pushDiagnostic(
              diagnostics,
              "error",
              "local",
              `${componentPath}.attention.n_head`,
              "n_embd must be divisible by n_head."
            );
          }
          if (
            Number.isInteger(config.n_embd) &&
            config.n_embd > 0 &&
            config.n_embd % n_head === 0 &&
            (config.n_embd / n_head) % 2 !== 0
          ) {
            pushDiagnostic(
              diagnostics,
              "error",
              "local",
              `${componentPath}.attention`,
              "Rotary embeddings require an even head_dim (n_embd / n_head)."
            );
          }
        }
      }

      if ("mlp" in component) {
        const { multiplier, sequence } = component.mlp;
        const mlpPath = `${componentPath}.mlp`;

        if (!Number.isFinite(multiplier) || multiplier <= 0) {
          pushDiagnostic(
            diagnostics,
            "error",
            "local",
            `${mlpPath}.multiplier`,
            "MLP multiplier must be a finite number greater than 0."
          );
        }

        if (!Array.isArray(sequence) || sequence.length === 0) {
          pushDiagnostic(
            diagnostics,
            "error",
            "local",
            `${mlpPath}.sequence`,
            "MLP sequence must contain at least one step."
          );
          return;
        }

        const linearIndices: number[] = [];
        let activationCount = 0;

        sequence.forEach((step, stepIndex) => {
          if ("linear" in step) {
            linearIndices.push(stepIndex);
          }
          if ("activation" in step) {
            activationCount += 1;
          }
          if ("norm" in step && step.norm.type === "rmsnorm") {
            if (typeof step.norm.learnable_gamma !== "boolean") {
              pushDiagnostic(
                diagnostics,
                "error",
                "local",
                `${mlpPath}.sequence[${stepIndex}].norm.learnable_gamma`,
                "RMSNorm requires learnable_gamma boolean."
              );
            }
          }
        });

        if (linearIndices.length === 0) {
          pushDiagnostic(
            diagnostics,
            "warning",
            "local",
            mlpPath,
            "MLP sequence has no linear steps. This behaves like a nonlinear pass-through."
          );
        } else {
          const firstLinearIndex = linearIndices[0];
          const lastLinearIndex = linearIndices[linearIndices.length - 1];
          const multiplierIsIdentity = Math.abs(multiplier - 1) < 1e-9;

          if (firstLinearIndex !== 0 && !multiplierIsIdentity) {
            pushDiagnostic(
              diagnostics,
              "error",
              "local",
              mlpPath,
              "When multiplier != 1, the first MLP step should be linear to preserve input dimensions."
            );
          }
          if (lastLinearIndex !== sequence.length - 1 && !multiplierIsIdentity) {
            pushDiagnostic(
              diagnostics,
              "error",
              "local",
              mlpPath,
              "When multiplier != 1, the last MLP step should be linear to project back to n_embd."
            );
          }
          if (linearIndices.length === 1 && !multiplierIsIdentity) {
            pushDiagnostic(
              diagnostics,
              "error",
              "local",
              mlpPath,
              "A single linear step with multiplier != 1 will leave the MLP output dimension mismatched."
            );
          }
        }

        if (activationCount === 0) {
          pushDiagnostic(
            diagnostics,
            "warning",
            "local",
            mlpPath,
            "MLP sequence has no activation step."
          );
        }
      }
    });
  });

  return diagnostics;
}

function parseActivationType(value: unknown, path: string, errors: string[]): ActivationType {
  if (typeof value === "string" && (ACTIVATION_TYPES as readonly string[]).includes(value)) {
    return value as ActivationType;
  }
  errors.push(`${path} must be one of: ${ACTIVATION_TYPES.join(", ")}.`);
  return "relu";
}

function parseIntegerField(value: unknown, path: string, errors: string[]): number {
  if (typeof value === "number" && Number.isFinite(value) && Number.isInteger(value)) {
    return value;
  }
  errors.push(`${path} must be an integer.`);
  return 0;
}

function parseNumberField(value: unknown, path: string, errors: string[]): number {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  errors.push(`${path} must be a number.`);
  return 0;
}

function parseBooleanField(value: unknown, path: string, errors: string[]): boolean {
  if (typeof value === "boolean") {
    return value;
  }
  errors.push(`${path} must be a boolean.`);
  return false;
}

function parseNormConfig(value: unknown, path: string, errors: string[]): NormConfig {
  if (!isRecord(value)) {
    errors.push(`${path} must be an object.`);
    return { type: "layernorm" };
  }
  if (value.type === "layernorm") {
    const out: LayerNormConfig = { type: "layernorm" };
    return out;
  }
  if (value.type === "rmsnorm") {
    const learnable_gamma = parseBooleanField(value.learnable_gamma, `${path}.learnable_gamma`, errors);
    const out: RMSNormConfig = { type: "rmsnorm", learnable_gamma };
    return out;
  }
  errors.push(`${path}.type must be 'layernorm' or 'rmsnorm'.`);
  return { type: "layernorm" };
}

function parseActivationComponent(
  value: unknown,
  path: string,
  errors: string[]
): ActivationComponentConfig {
  if (!isRecord(value) || !isRecord(value.activation)) {
    errors.push(`${path}.activation must be an object.`);
    return { activation: { type: "relu" } };
  }
  return {
    activation: {
      type: parseActivationType(value.activation.type, `${path}.activation.type`, errors),
    },
  };
}

function parseNormComponent(value: unknown, path: string, errors: string[]): NormComponentConfig {
  if (!isRecord(value) || !("norm" in value)) {
    errors.push(`${path}.norm must be present.`);
    return { norm: { type: "layernorm" } };
  }
  return { norm: parseNormConfig(value.norm, `${path}.norm`, errors) };
}

function parseMlpStep(value: unknown, path: string, errors: string[]): MlpStep {
  if (!isRecord(value)) {
    errors.push(`${path} must be an object.`);
    return { activation: { type: "relu" } };
  }
  const keys = ["linear", "norm", "activation"].filter((key) => key in value);
  if (keys.length !== 1) {
    errors.push(`${path} must contain exactly one of linear/norm/activation.`);
    return { activation: { type: "relu" } };
  }

  if ("linear" in value) {
    if (!isRecord(value.linear)) {
      errors.push(`${path}.linear must be an object.`);
      return { linear: { bias: true } };
    }
    return {
      linear: {
        bias: parseBooleanField(value.linear.bias, `${path}.linear.bias`, errors),
      },
    };
  }
  if ("norm" in value) {
    return { norm: parseNormConfig(value.norm, `${path}.norm`, errors) };
  }
  return {
    activation: {
      type: parseActivationType(
        isRecord(value.activation) ? value.activation.type : undefined,
        `${path}.activation.type`,
        errors
      ),
    },
  };
}

function parseBlockComponent(value: unknown, path: string, errors: string[]): BlockComponent {
  if (!isRecord(value)) {
    errors.push(`${path} must be an object.`);
    return { activation: { type: "relu" } };
  }

  const keys = ["attention", "mlp", "norm", "activation"].filter((key) => key in value);
  if (keys.length !== 1) {
    errors.push(`${path} must contain exactly one of attention/mlp/norm/activation.`);
    return { activation: { type: "relu" } };
  }

  if ("attention" in value) {
    if (!isRecord(value.attention)) {
      errors.push(`${path}.attention must be an object.`);
      return { attention: { n_head: 12, n_kv_head: 12 } };
    }
    return {
      attention: {
        n_head: parseIntegerField(value.attention.n_head, `${path}.attention.n_head`, errors),
        n_kv_head: parseIntegerField(value.attention.n_kv_head, `${path}.attention.n_kv_head`, errors),
      },
    };
  }

  if ("mlp" in value) {
    if (!isRecord(value.mlp)) {
      errors.push(`${path}.mlp must be an object.`);
      return { mlp: { multiplier: 4, sequence: [] } };
    }
    const sequenceValue = value.mlp.sequence;
    const sequence: MlpStep[] = [];
    if (!Array.isArray(sequenceValue)) {
      errors.push(`${path}.mlp.sequence must be an array.`);
    } else {
      sequenceValue.forEach((step, stepIndex) => {
        sequence.push(parseMlpStep(step, `${path}.mlp.sequence[${stepIndex}]`, errors));
      });
    }
    return {
      mlp: {
        multiplier: parseNumberField(value.mlp.multiplier, `${path}.mlp.multiplier`, errors),
        sequence,
      },
    };
  }

  if ("norm" in value) {
    return parseNormComponent(value, path, errors);
  }

  return parseActivationComponent(value, path, errors);
}

export function parseImportedModelConfig(value: unknown): ImportedModelConfigParseResult {
  const errors: string[] = [];
  if (!isRecord(value)) {
    return { config: null, errors: ["Root JSON value must be an object."] };
  }

  const blocksValue = value.blocks;
  const blocks: ModelBlock[] = [];
  if (!Array.isArray(blocksValue)) {
    errors.push("blocks must be an array.");
  } else {
    blocksValue.forEach((blockValue, blockIndex) => {
      const blockPath = `blocks[${blockIndex}]`;
      if (!isRecord(blockValue)) {
        errors.push(`${blockPath} must be an object.`);
        return;
      }
      const componentsValue = blockValue.components;
      if (!Array.isArray(componentsValue)) {
        errors.push(`${blockPath}.components must be an array.`);
        blocks.push({ components: [] });
        return;
      }
      const components = componentsValue.map((component, componentIndex) =>
        parseBlockComponent(component, `${blockPath}.components[${componentIndex}]`, errors)
      );
      blocks.push({ components });
    });
  }

  const config: ModelConfig = {
    context_length: parseIntegerField(value.context_length, "context_length", errors),
    vocab_size: parseIntegerField(value.vocab_size, "vocab_size", errors),
    n_embd: parseIntegerField(value.n_embd, "n_embd", errors),
    weight_tying: parseBooleanField(value.weight_tying, "weight_tying", errors),
    blocks,
  };

  return {
    config: errors.length === 0 ? config : null,
    errors,
  };
}
