import { ConfigNumberInput } from "../../shared/components/ConfigNumberInput";
import {
  asNumber,
  asRecord,
  asString,
} from "../lib/object";

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
      <summary>Advanced runtime controls</summary>
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
              <span>Beginning-of-sequence token</span>
              <input
                value={asString(dataloaderConfig.bos_token)}
                onChange={(event) =>
                  handleDataloaderField(["bos_token"], event.target.value)
                }
              />
            </label>
            <label className="fieldLabel">
              <span>End-of-sequence token</span>
              <input
                value={asString(dataloaderConfig.eos_token)}
                onChange={(event) =>
                  handleDataloaderField(["eos_token"], event.target.value)
                }
              />
            </label>
            <label className="fieldLabel">
              <span>Padding token</span>
              <input
                value={asString(dataloaderConfig.pad_token)}
                onChange={(event) =>
                  handleDataloaderField(["pad_token"], event.target.value)
                }
              />
            </label>
            <label className="fieldLabel">
              <span>Token data type</span>
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
              <span>Pretokenize batch size</span>
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
              <span>Cache directory</span>
              <input
                value={asString(dataloaderConfig.cache_dir)}
                onChange={(event) =>
                  handleDataloaderField(["cache_dir"], event.target.value)
                }
              />
            </label>
            <label className="fieldLabel">
              <span>Optimizer betas</span>
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
              <span>Optimizer epsilon</span>
              <ConfigNumberInput
                mode="decimal"
                step="0.00000001"
                value={asNumber(asRecord(trainingConfig.optimizer).eps, 1e-8)}
                onCommit={(value) => handleTrainingField(["optimizer", "eps"], value)}
              />
            </label>
            <label className="fieldLabel">
              <span>Distributed node split</span>
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
              <span>Distributed node rank</span>
              <ConfigNumberInput
                value={asNumber(dataloaderConfig.node_rank, 0)}
                onCommit={(value) => handleDataloaderField(["node_rank"], value)}
              />
            </label>
            <label className="fieldLabel">
              <span>Distributed node world size</span>
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
