import { ConfigNumberInput } from "../../shared/components/ConfigNumberInput";
import {
  asNumber,
  asRecord,
  asString,
} from "../lib/object";
import { FieldLabelText, InfoTooltip } from "../../shared/components/HelpTooltip";

interface AdvancedRuntimePanelProps {
  dataloaderConfig: Record<string, unknown>;
  handleDataloaderField: (path: string[], value: unknown) => void;
  handleTrainingField: (path: string[], value: unknown) => void;
  trainingConfig: Record<string, unknown>;
}

export function AdvancedRuntimePanel({
  dataloaderConfig,
  handleDataloaderField,
  handleTrainingField,
  trainingConfig,
}: AdvancedRuntimePanelProps) {
  return (
    <details className="settingsPanel">
      <summary>
        <span>Advanced runtime controls</span>
        <InfoTooltip label="Advanced runtime controls explanation" align="right" width="wide">
          <strong>Advanced runtime controls</strong>
          <p>
            These values map directly to tokenizer/dataloader and optimizer internals. Most
            users should keep defaults unless matching an existing training recipe.
          </p>
        </InfoTooltip>
      </summary>
      <div className="settingsGrid">
        <div className="settingsGroup">
          <div className="settingsGroupHeader">
            <h3>Deeper runtime options</h3>
            <p className="settingsGroupHint">
              Token formatting, optimizer internals, and multi-node controls stay available
              without taking over the main workflow.
            </p>
          </div>
          <div className="fieldGrid trainingSettingsCompactGrid">
            <label className="fieldLabel">
              <FieldLabelText tooltipLabel="BOS token explanation" tooltip="Token inserted at the beginning of sequences when the dataloader prepares text. It must exist in the selected tokenizer vocabulary.">
                Beginning-of-sequence token
              </FieldLabelText>
              <input
                value={asString(dataloaderConfig.bos_token)}
                onChange={(event) =>
                  handleDataloaderField(["bos_token"], event.target.value)
                }
              />
            </label>
            <label className="fieldLabel">
              <FieldLabelText tooltipLabel="EOS token explanation" tooltip="Token used to mark the end of a sequence. It helps the model learn where documents or samples finish.">
                End-of-sequence token
              </FieldLabelText>
              <input
                value={asString(dataloaderConfig.eos_token)}
                onChange={(event) =>
                  handleDataloaderField(["eos_token"], event.target.value)
                }
              />
            </label>
            <label className="fieldLabel">
              <FieldLabelText tooltipLabel="Padding token explanation" tooltip="Token used when examples need padding to the same length. It must be a token the tokenizer can encode.">
                Padding token
              </FieldLabelText>
              <input
                value={asString(dataloaderConfig.pad_token)}
                onChange={(event) =>
                  handleDataloaderField(["pad_token"], event.target.value)
                }
              />
            </label>
            <label className="fieldLabel">
              <FieldLabelText tooltipLabel="Token data type explanation" tooltip="Numeric type used to store token IDs. int64 is safest; smaller types save memory only when vocabulary IDs fit inside the type range.">
                Token data type
              </FieldLabelText>
              <select
                value={asString(dataloaderConfig.token_dtype, "int64")}
                onChange={(event) =>
                  handleDataloaderField(["token_dtype"], event.target.value)
                }
              >
                <option value="int64">int64</option>
                <option value="int32">int32</option>
                <option value="int16">int16</option>
                <option value="uint8">uint8</option>
              </select>
            </label>
            <label className="fieldLabel">
              <FieldLabelText tooltipLabel="Pretokenize batch size explanation" tooltip="How many records the dataloader tokenizes at a time before training. Larger batches can improve throughput but use more memory.">
                Pretokenize batch size
              </FieldLabelText>
              <ConfigNumberInput
                value={asNumber(dataloaderConfig.pretokenize_batch_size, 1000)}
                onCommit={(value) =>
                  handleDataloaderField(
                    ["pretokenize_batch_size"],
                    value
                  )
                }
              />
            </label>
            <label className="fieldLabel">
              <FieldLabelText tooltipLabel="Cache directory explanation" tooltip="Optional directory for cached tokenized data or dataset artifacts. Leave blank to let the backend choose its default cache location.">
                Cache directory
              </FieldLabelText>
              <input
                value={asString(dataloaderConfig.cache_dir)}
                onChange={(event) =>
                  handleDataloaderField(["cache_dir"], event.target.value)
                }
              />
            </label>
            <label className="fieldLabel">
              <FieldLabelText tooltipLabel="Optimizer betas explanation" tooltip="AdamW momentum coefficients as two comma-separated numbers. The first smooths gradients; the second smooths squared gradients.">
                Optimizer betas
              </FieldLabelText>
              <input
                value={
                  Array.isArray(asRecord(trainingConfig.optimizer).betas)
                    ? (asRecord(trainingConfig.optimizer).betas as unknown[])
                        .map(String)
                        .join(", ")
                    : "0.9, 0.95"
                }
                onChange={(event) =>
                  handleTrainingField(
                    ["optimizer", "betas"],
                    event.target.value
                      .split(",")
                      .map((item) => Number(item.trim()))
                      .filter((value) => Number.isFinite(value))
                  )
                }
              />
            </label>
            <label className="fieldLabel">
              <FieldLabelText tooltipLabel="Optimizer epsilon explanation" tooltip="Small stabilizing value used by AdamW to avoid division by zero. Changing it is rarely needed.">
                Optimizer epsilon
              </FieldLabelText>
              <ConfigNumberInput
                mode="decimal"
                step="0.00000001"
                value={asNumber(asRecord(trainingConfig.optimizer).eps, 1e-8)}
                onCommit={(value) => handleTrainingField(["optimizer", "eps"], value)}
              />
            </label>
            <label className="fieldLabel">
              <FieldLabelText tooltipLabel="Distributed node split explanation" tooltip="Splits dataset records by node rank for multi-node training so each node reads a different shard. Keep disabled for local or single-pod runs.">
                Distributed node split
              </FieldLabelText>
              <select
                value={String(Boolean(dataloaderConfig.node_split))}
                onChange={(event) =>
                  handleDataloaderField(
                    ["node_split"],
                    event.target.value === "true"
                  )
                }
              >
                <option value="false">Disabled</option>
                <option value="true">Enabled</option>
              </select>
            </label>
            <label className="fieldLabel">
              <FieldLabelText tooltipLabel="Distributed node rank explanation" tooltip="Zero-based index of this node in a multi-node run. It is ignored unless distributed node split is enabled.">
                Distributed node rank
              </FieldLabelText>
              <ConfigNumberInput
                value={asNumber(dataloaderConfig.node_rank, 0)}
                onCommit={(value) => handleDataloaderField(["node_rank"], value)}
              />
            </label>
            <label className="fieldLabel">
              <FieldLabelText tooltipLabel="Distributed node world size explanation" tooltip="Total number of nodes participating in a distributed run. It is used with node rank to split records.">
                Distributed node world size
              </FieldLabelText>
              <ConfigNumberInput
                value={asNumber(dataloaderConfig.node_world_size, 1)}
                onCommit={(value) => handleDataloaderField(["node_world_size"], value)}
              />
            </label>
          </div>
        </div>
      </div>
    </details>
  );
}
