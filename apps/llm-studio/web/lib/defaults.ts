export const ACTIVATION_TYPES = [
  "gelu",
  "relu",
  "squared_relu",
  "silu",
  "tanh",
  "sigmoid",
] as const;

export type ActivationType = (typeof ACTIVATION_TYPES)[number];

export type RMSNormConfig = {
  type: "rmsnorm";
  learnable_gamma: boolean;
};

export type LayerNormConfig = {
  type: "layernorm";
};

export type NormConfig = RMSNormConfig | LayerNormConfig;

export interface AttentionConfig {
  n_head: number;
  n_kv_head: number;
}

export interface LinearConfig {
  bias: boolean;
}

export interface ActivationConfig {
  type: ActivationType;
}

export interface MlpConfig {
  multiplier: number;
  sequence: MlpStep[];
}

export interface AttentionComponentConfig {
  attention: AttentionConfig;
}

export interface MlpComponentConfig {
  mlp: MlpConfig;
}

export interface NormComponentConfig {
  norm: NormConfig;
}

export interface ActivationComponentConfig {
  activation: ActivationConfig;
}

export interface LinearStepConfig {
  linear: LinearConfig;
}

export type MlpStep =
  | LinearStepConfig
  | NormComponentConfig
  | ActivationComponentConfig;

export type BlockComponent =
  | AttentionComponentConfig
  | MlpComponentConfig
  | NormComponentConfig
  | ActivationComponentConfig;

export interface ModelBlock {
  components: BlockComponent[];
}

export interface ModelConfig {
  context_length: number;
  vocab_size: number;
  n_embd: number;
  weight_tying: boolean;
  blocks: ModelBlock[];
}

export function createDefaultNormConfig(): NormConfig {
  return { type: "layernorm" };
}

export function createDefaultMlpSequence(): MlpStep[] {
  return [
    { linear: { bias: true } },
    { activation: { type: "relu" } },
    { linear: { bias: true } },
  ];
}

export function createDefaultBlockConfig(): ModelBlock {
  return {
    components: [
      { norm: createDefaultNormConfig() },
      { attention: { n_head: 12, n_kv_head: 12 } },
      { norm: createDefaultNormConfig() },
      {
        mlp: {
          multiplier: 4,
          sequence: createDefaultMlpSequence(),
        },
      },
    ],
  };
}

export function createDefaultModelConfig(blockCount = 6): ModelConfig {
  return {
    context_length: 1024,
    vocab_size: 1000,
    n_embd: 768,
    weight_tying: true,
    blocks: Array.from({ length: blockCount }, () => createDefaultBlockConfig()),
  };
}

export const defaultModelConfig: ModelConfig = createDefaultModelConfig();
