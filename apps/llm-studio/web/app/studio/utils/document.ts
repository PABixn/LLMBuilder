import type {
  ActivationType,
  ActivationComponentConfig,
  AttentionComponentConfig,
  BlockComponent,
  LinearStepConfig,
  MlpComponentConfig,
  MlpStep,
  ModelBlock,
  ModelConfig,
  NormComponentConfig,
  NormConfig,
} from "../../../lib/defaults";
import type {
  BuilderMetrics,
  ConsecutiveBlockGroup,
  MlpComponentRef,
  MlpStepKind,
  StudioBlock,
  StudioComponent,
  StudioComponentKind,
  StudioComponentPrefab,
  StudioDocument,
  StudioMlpStep,
} from "../types";

export function createId(prefix: string): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `${prefix}-${crypto.randomUUID()}`;
  }
  return `${prefix}-${Math.random().toString(36).slice(2, 10)}`;
}

export function clone<T>(value: T): T {
  if (typeof structuredClone === "function") {
    return structuredClone(value);
  }
  return JSON.parse(JSON.stringify(value)) as T;
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

export function moveItem<T>(items: T[], fromIndex: number, toIndex: number): T[] {
  if (
    fromIndex < 0 ||
    fromIndex >= items.length ||
    toIndex < 0 ||
    toIndex > items.length ||
    (fromIndex === toIndex || fromIndex + 1 === toIndex)
  ) {
    return items.slice();
  }
  const next = items.slice();
  const [item] = next.splice(fromIndex, 1);
  const adjustedIndex = fromIndex < toIndex ? toIndex - 1 : toIndex;
  next.splice(adjustedIndex, 0, item);
  return next;
}

export function createDefaultStudioNorm(): NormConfig {
  return { type: "layernorm" };
}

export function createDefaultStudioMlpStep(kind: MlpStepKind): StudioMlpStep {
  if (kind === "linear") {
    return { id: createId("mlp-step"), kind: "linear", linear: { bias: true } };
  }
  if (kind === "norm") {
    return { id: createId("mlp-step"), kind: "norm", norm: createDefaultStudioNorm() };
  }
  return {
    id: createId("mlp-step"),
    kind: "activation",
    activation: { type: "relu" },
  };
}

export function createDefaultStudioComponent(kind: StudioComponentKind): StudioComponent {
  if (kind === "attention") {
    return {
      id: createId("component"),
      kind: "attention",
      attention: { n_head: 12, n_kv_head: 12 },
    };
  }
  if (kind === "mlp") {
    return {
      id: createId("component"),
      kind: "mlp",
      mlp: {
        multiplier: 4,
        sequence: [
          createDefaultStudioMlpStep("linear"),
          createDefaultStudioMlpStep("activation"),
          createDefaultStudioMlpStep("linear"),
        ],
      },
    };
  }
  if (kind === "norm") {
    return { id: createId("component"), kind: "norm", norm: createDefaultStudioNorm() };
  }
  return {
    id: createId("component"),
    kind: "activation",
    activation: { type: "relu" },
  };
}

export function studioMlpStepFromConfig(step: MlpStep): StudioMlpStep {
  if ("linear" in step) {
    return {
      id: createId("mlp-step"),
      kind: "linear",
      linear: clone(step.linear),
    };
  }
  if ("norm" in step) {
    return {
      id: createId("mlp-step"),
      kind: "norm",
      norm: clone(step.norm),
    };
  }
  return {
    id: createId("mlp-step"),
    kind: "activation",
    activation: clone(step.activation),
  };
}

export function studioComponentFromConfig(component: BlockComponent): StudioComponent {
  if ("attention" in component) {
    return {
      id: createId("component"),
      kind: "attention",
      attention: clone(component.attention),
    };
  }
  if ("mlp" in component) {
    return {
      id: createId("component"),
      kind: "mlp",
      mlp: {
        multiplier: component.mlp.multiplier,
        sequence: component.mlp.sequence.map(studioMlpStepFromConfig),
      },
    };
  }
  if ("norm" in component) {
    return {
      id: createId("component"),
      kind: "norm",
      norm: clone(component.norm),
    };
  }
  return {
    id: createId("component"),
    kind: "activation",
    activation: clone(component.activation),
  };
}

export function studioBlockFromConfig(block: ModelBlock): StudioBlock {
  return {
    id: createId("block"),
    components: block.components.map(studioComponentFromConfig),
  };
}

export function studioDocumentFromConfig(config: ModelConfig): StudioDocument {
  return {
    context_length: config.context_length,
    vocab_size: config.vocab_size,
    n_embd: config.n_embd,
    weight_tying: config.weight_tying,
    blocks: config.blocks.map(studioBlockFromConfig),
  };
}

export function mlpStepToConfig(step: StudioMlpStep): MlpStep {
  if (step.kind === "linear") {
    const out: LinearStepConfig = { linear: clone(step.linear) };
    return out;
  }
  if (step.kind === "norm") {
    const out: NormComponentConfig = { norm: clone(step.norm) };
    return out;
  }
  const out: ActivationComponentConfig = { activation: clone(step.activation) };
  return out;
}

export function studioComponentToConfig(component: StudioComponent): BlockComponent {
  if (component.kind === "attention") {
    const out: AttentionComponentConfig = { attention: clone(component.attention) };
    return out;
  }
  if (component.kind === "mlp") {
    const out: MlpComponentConfig = {
      mlp: {
        multiplier: component.mlp.multiplier,
        sequence: component.mlp.sequence.map(mlpStepToConfig),
      },
    };
    return out;
  }
  if (component.kind === "norm") {
    const out: NormComponentConfig = { norm: clone(component.norm) };
    return out;
  }
  const out: ActivationComponentConfig = { activation: clone(component.activation) };
  return out;
}

export function studioComponentKindFromConfig(component: BlockComponent): StudioComponentKind {
  if ("attention" in component) {
    return "attention";
  }
  if ("mlp" in component) {
    return "mlp";
  }
  if ("norm" in component) {
    return "norm";
  }
  return "activation";
}

export function createComponentPrefab(
  component: StudioComponent,
  name: string
): StudioComponentPrefab {
  return {
    id: createId("prefab"),
    name,
    kind: component.kind,
    component: studioComponentToConfig(component),
    createdAt: Date.now(),
  };
}

export function instantiateComponentFromPrefab(prefab: StudioComponentPrefab): StudioComponent {
  return studioComponentFromConfig(prefab.component);
}

export function studioBlockToConfig(block: StudioBlock): ModelBlock {
  return {
    components: block.components.map(studioComponentToConfig),
  };
}

export function blockConfigSignature(block: StudioBlock): string {
  return JSON.stringify(studioBlockToConfig(block));
}

export function collectConsecutiveIdenticalBlockGroups(blocks: StudioBlock[]): ConsecutiveBlockGroup[] {
  if (blocks.length === 0) {
    return [];
  }

  const signatures = blocks.map(blockConfigSignature);
  const groups: ConsecutiveBlockGroup[] = [];

  let startIndex = 0;
  while (startIndex < blocks.length) {
    const signature = signatures[startIndex];
    let endIndex = startIndex;
    while (endIndex + 1 < blocks.length && signatures[endIndex + 1] === signature) {
      endIndex += 1;
    }

    groups.push({
      key: `block-group:${blocks[startIndex].id}:${blocks[endIndex].id}:${endIndex - startIndex + 1}`,
      startIndex,
      endIndex,
      count: endIndex - startIndex + 1,
    });

    startIndex = endIndex + 1;
  }

  return groups;
}

export function studioDocumentToConfig(document: StudioDocument): ModelConfig {
  return {
    context_length: document.context_length,
    vocab_size: document.vocab_size,
    n_embd: document.n_embd,
    weight_tying: document.weight_tying,
    blocks: document.blocks.map(studioBlockToConfig),
  };
}

export function cloneBlockWithNewIds(block: StudioBlock): StudioBlock {
  return {
    id: createId("block"),
    components: block.components.map((component) => {
      if (component.kind === "mlp") {
        return {
          ...clone(component),
          id: createId("component"),
          mlp: {
            multiplier: component.mlp.multiplier,
            sequence: component.mlp.sequence.map((step) => ({
              ...clone(step),
              id: createId("mlp-step"),
            })),
          },
        } satisfies StudioComponent;
      }
      return {
        ...clone(component),
        id: createId("component"),
      } satisfies StudioComponent;
    }),
  };
}

export function findBlockIndex(document: StudioDocument, blockId: string): number {
  return document.blocks.findIndex((block) => block.id === blockId);
}

export function findComponentIndex(block: StudioBlock, componentId: string): number {
  return block.components.findIndex((component) => component.id === componentId);
}

export function getMlpComponent(
  document: StudioDocument,
  blockId: string,
  componentId: string
): MlpComponentRef | null {
  const blockIndex = findBlockIndex(document, blockId);
  if (blockIndex < 0) {
    return null;
  }
  const componentIndex = findComponentIndex(document.blocks[blockIndex], componentId);
  if (componentIndex < 0) {
    return null;
  }
  const component = document.blocks[blockIndex].components[componentIndex];
  if (component.kind !== "mlp") {
    return null;
  }
  return { blockIndex, componentIndex, component };
}

export function labelForComponentKind(kind: StudioComponentKind): string {
  if (kind === "attention") {
    return "Attention";
  }
  if (kind === "mlp") {
    return "MLP";
  }
  if (kind === "norm") {
    return "Norm";
  }
  return "Activation";
}

export function labelForMlpStepKind(kind: MlpStepKind): string {
  if (kind === "linear") {
    return "Linear";
  }
  if (kind === "norm") {
    return "Norm";
  }
  return "Activation";
}

export function labelForNormType(type: NormConfig["type"]): string {
  if (type === "layernorm") {
    return "LayerNorm";
  }
  return "RMSNorm";
}

export function labelForActivationType(type: ActivationType): string {
  if (type === "gelu") {
    return "GELU";
  }
  if (type === "relu") {
    return "ReLU";
  }
  if (type === "squared_relu") {
    return "Squared ReLU";
  }
  if (type === "silu") {
    return "SiLU";
  }
  if (type === "tanh") {
    return "Tanh";
  }
  return "Sigmoid";
}

export function summarizeComponent(component: StudioComponent): string {
  if (component.kind === "attention") {
    return `${component.attention.n_head} heads / ${component.attention.n_kv_head} kv`;
  }
  if (component.kind === "mlp") {
    return `${component.mlp.sequence.length} steps, x${component.mlp.multiplier}`;
  }
  if (component.kind === "norm") {
    if (component.norm.type === "layernorm") {
      return labelForNormType(component.norm.type);
    }
    return component.norm.learnable_gamma ? "RMSNorm (learnable)" : "RMSNorm (fixed)";
  }
  return labelForActivationType(component.activation.type);
}

export function summarizeComponentConfig(component: BlockComponent): string {
  if ("attention" in component) {
    return `${component.attention.n_head} heads / ${component.attention.n_kv_head} kv`;
  }
  if ("mlp" in component) {
    return `${component.mlp.sequence.length} steps, x${component.mlp.multiplier}`;
  }
  if ("norm" in component) {
    if (component.norm.type === "layernorm") {
      return labelForNormType(component.norm.type);
    }
    return component.norm.learnable_gamma ? "RMSNorm (learnable)" : "RMSNorm (fixed)";
  }
  return labelForActivationType(component.activation.type);
}

export function summarizeMlpStep(step: StudioMlpStep): string {
  if (step.kind === "linear") {
    return step.linear.bias ? "Bias on" : "Bias off";
  }
  if (step.kind === "norm") {
    if (step.norm.type === "layernorm") {
      return labelForNormType(step.norm.type);
    }
    return step.norm.learnable_gamma ? "RMSNorm learnable" : "RMSNorm fixed";
  }
  return labelForActivationType(step.activation.type);
}

export function collectBuilderMetrics(document: StudioDocument): BuilderMetrics {
  const metrics: BuilderMetrics = {
    blockCount: document.blocks.length,
    componentCount: 0,
    attentionCount: 0,
    mlpCount: 0,
    normCount: 0,
    activationCount: 0,
    mlpStepCount: 0,
    mlpActivationStepCount: 0,
  };

  for (const block of document.blocks) {
    metrics.componentCount += block.components.length;
    for (const component of block.components) {
      if (component.kind === "attention") {
        metrics.attentionCount += 1;
      } else if (component.kind === "mlp") {
        metrics.mlpCount += 1;
        metrics.mlpStepCount += component.mlp.sequence.length;
        for (const step of component.mlp.sequence) {
          if (step.kind === "activation") {
            metrics.mlpActivationStepCount += 1;
          }
        }
      } else if (component.kind === "norm") {
        metrics.normCount += 1;
      } else if (component.kind === "activation") {
        metrics.activationCount += 1;
      }
    }
  }

  return metrics;
}

export function collectAllComponentIds(document: StudioDocument): string[] {
  const ids: string[] = [];
  for (const block of document.blocks) {
    for (const component of block.components) {
      ids.push(component.id);
    }
  }
  return ids;
}

export function collectAllMlpStepIds(document: StudioDocument): string[] {
  const ids: string[] = [];
  for (const block of document.blocks) {
    for (const component of block.components) {
      if (component.kind !== "mlp") {
        continue;
      }
      for (const step of component.mlp.sequence) {
        ids.push(step.id);
      }
    }
  }
  return ids;
}
