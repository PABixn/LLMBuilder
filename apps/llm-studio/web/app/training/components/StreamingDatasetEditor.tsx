import { FiTrash2 } from "react-icons/fi";

import {
  FILTER_OPERATORS,
} from "../constants";
import type {
  FilterOperator,
  StreamingDatasetFormState,
  StreamingFilterFormState,
} from "../types";
import { FieldLabelText, HelpTooltip, InfoTooltip } from "../../shared/components/HelpTooltip";

interface StreamingDatasetEditorProps {
  addStreamingDataset: () => void;
  addStreamingFilter: (datasetId: string) => void;
  handleLoadStreamingTemplate: () => void;
  hfToken: string;
  isLoadingDatasetTemplate: boolean;
  removeStreamingDataset: (datasetId: string) => void;
  removeStreamingFilter: (datasetId: string, filterId: string) => void;
  setHfToken: (value: string) => void;
  streamingDatasets: StreamingDatasetFormState[];
  updateStreamingDataset: (
    datasetId: string,
    updates: Partial<Omit<StreamingDatasetFormState, "id">>
  ) => void;
  updateStreamingFilter: (
    datasetId: string,
    filterId: string,
    updates: Partial<Omit<StreamingFilterFormState, "id">>
  ) => void;
  updateStreamingWeight: (datasetId: string, rawWeight: string) => void;
}

export function StreamingDatasetEditor({
  addStreamingDataset,
  addStreamingFilter,
  handleLoadStreamingTemplate,
  hfToken,
  isLoadingDatasetTemplate,
  removeStreamingDataset,
  removeStreamingFilter,
  setHfToken,
  streamingDatasets,
  updateStreamingDataset,
  updateStreamingFilter,
  updateStreamingWeight,
}: StreamingDatasetEditorProps) {
  return (
    <div className="datasetConfigurator trainingTokenizerDatasetSection">
      <label className="fieldLabel fullWidthField">
        <FieldLabelText tooltipLabel="HF token explanation" tooltip="Optional Hugging Face access token. Use it only when a private or gated dataset needs authentication; public datasets do not need it.">
          HF token <small>optional</small>
        </FieldLabelText>
        <input
          type="password"
          value={hfToken}
          onChange={(event) => setHfToken(event.target.value)}
          autoComplete="off"
          placeholder="hf_..."
        />
        <span className="fieldNote">Needed for private datasets.</span>
      </label>

      <div className="actionRow">
        <HelpTooltip label="Add streaming dataset" content="Adds another dataset entry to the mixture. Dataset weights are normalized so the combined stream samples from each source in proportion to its weight.">
          <button
            type="button"
            className="secondaryButton"
            onClick={addStreamingDataset}
          >
            Add dataset
          </button>
        </HelpTooltip>
        <HelpTooltip label="Load streaming dataset template" content="Loads the backend dataloader template for streaming datasets and switches this panel to the template defaults.">
          <button
            type="button"
            className="secondaryButton"
            onClick={handleLoadStreamingTemplate}
            disabled={isLoadingDatasetTemplate}
          >
            {isLoadingDatasetTemplate
              ? "Loading template..."
              : "Load template"}
          </button>
        </HelpTooltip>
      </div>

      <div className="datasetList">
        {streamingDatasets.map((entry, index) => (
          <div key={entry.id} className="datasetCard">
            <div className="datasetCardHeader">
              <strong>Dataset {index + 1}</strong>
              <button
                type="button"
                className="textButton datasetRemoveButton"
                onClick={() => removeStreamingDataset(entry.id)}
                disabled={streamingDatasets.length <= 1}
                aria-label={`Remove streaming dataset ${index + 1}`}
                title={`Remove streaming dataset ${index + 1}`}
              >
                <FiTrash2 aria-hidden="true" />
              </button>
            </div>

            <div className="fieldGrid">
              <label className="fieldLabel">
                <FieldLabelText tooltipLabel="Dataset name explanation" tooltip="Hugging Face dataset repository name, for example owner/dataset. This is passed to the streaming dataset loader.">
                  Dataset name
                </FieldLabelText>
                <input
                  value={entry.name}
                  onChange={(event) =>
                    updateStreamingDataset(entry.id, {
                      name: event.target.value,
                    })
                  }
                  placeholder="HuggingFaceFW/fineweb-edu"
                />
              </label>

              <label className="fieldLabel">
                <FieldLabelText tooltipLabel="Dataset split explanation" tooltip="The dataset split to stream, usually train. Use the exact split name exposed by the Hugging Face dataset.">
                  Split
                </FieldLabelText>
                <input
                  value={entry.split}
                  onChange={(event) =>
                    updateStreamingDataset(entry.id, {
                      split: event.target.value,
                    })
                  }
                  placeholder="train"
                />
              </label>

              <label className="fieldLabel">
                <FieldLabelText tooltipLabel="Dataset weight explanation" tooltip="Relative share of this dataset in the mixture. A dataset with weight 2 is sampled about twice as often as one with weight 1.">
                  Weight
                </FieldLabelText>
                <input
                  inputMode="decimal"
                  pattern="[0-9]*[.]?[0-9]*"
                  min="0"
                  max="1"
                  step="0.000001"
                  value={entry.weight}
                  onChange={(event) =>
                    updateStreamingWeight(entry.id, event.target.value)
                  }
                  placeholder="1.0"
                />
              </label>

              <label className="fieldLabel fullWidthField">
                <FieldLabelText tooltipLabel="Text columns explanation" tooltip="Column names that contain text to train on. Use commas for multiple text columns; the app passes these to the dataloader.">
                  Text columns
                </FieldLabelText>
                <input
                  value={entry.textColumns}
                  onChange={(event) =>
                    updateStreamingDataset(entry.id, {
                      textColumns: event.target.value,
                    })
                  }
                  placeholder="text"
                />
              </label>
            </div>

            <details className="subPanel">
              <summary>
                <span>Advanced options</span>
                <InfoTooltip label="Streaming advanced options explanation" align="right" width="wide">
                  <p>
                    Use dataset configs and filters when a Hugging Face dataset has named
                    subsets or when only some records should enter training.
                  </p>
                </InfoTooltip>
              </summary>
              <div className="fieldGrid">
                <label className="fieldLabel">
                  <FieldLabelText tooltipLabel="Dataset config explanation" tooltip="Optional Hugging Face config/subset name, such as a language or corpus variant. Leave blank when the dataset has no named config.">
                    Dataset config <small>optional</small>
                  </FieldLabelText>
                  <input
                    value={entry.config}
                    onChange={(event) =>
                      updateStreamingDataset(entry.id, {
                        config: event.target.value,
                      })
                    }
                  />
                </label>

                <div className="fullWidthField filterBuilder">
                  <div className="filterBuilderHeader">
                    <span className="filterBuilderTitle">
                      Filters (optional)
                      <InfoTooltip label="Streaming filters explanation" align="left" width="wide">
                        <p>
                          Filters keep or remove records before text is used. Column, operator,
                          and value become dataloader filter rules for each streamed record.
                        </p>
                      </InfoTooltip>
                    </span>
                    <button
                      type="button"
                      className="secondaryButton"
                      onClick={() => addStreamingFilter(entry.id)}
                    >
                      Add filter
                    </button>
                  </div>

                  {entry.filters.length === 0 ? (
                    <p className="filterEmpty">No filters yet.</p>
                  ) : (
                    <div className="filterList">
                      {entry.filters.map((filter) => (
                        <div key={filter.id} className="filterRow">
                          <label className="fieldLabel">
                            <FieldLabelText tooltipLabel="Filter column explanation" tooltip="Dataset column to inspect for this filter, such as language, language_score, or quality_score.">
                              Column
                            </FieldLabelText>
                            <input
                              value={filter.column}
                              onChange={(event) =>
                                updateStreamingFilter(entry.id, filter.id, {
                                  column: event.target.value,
                                })
                              }
                              placeholder="language_score"
                            />
                          </label>

                          <label className="fieldLabel">
                            <FieldLabelText tooltipLabel="Filter operator explanation" tooltip="Comparison used by the dataloader. For in and not in, the value can be JSON array syntax or comma-separated values.">
                              Operator
                            </FieldLabelText>
                            <select
                              value={filter.operator}
                              onChange={(event) =>
                                updateStreamingFilter(entry.id, filter.id, {
                                  operator: event.target
                                    .value as FilterOperator,
                                })
                              }
                            >
                              {FILTER_OPERATORS.map((operator) => (
                                <option key={operator} value={operator}>
                                  {operator}
                                </option>
                              ))}
                            </select>
                          </label>

                          <label className="fieldLabel">
                            <FieldLabelText tooltipLabel="Filter value explanation" tooltip="The value to compare against. The app detects booleans, numbers, JSON objects, JSON arrays, and comma-separated lists.">
                              Value
                            </FieldLabelText>
                            <input
                              value={filter.value}
                              onChange={(event) =>
                                updateStreamingFilter(entry.id, filter.id, {
                                  value: event.target.value,
                                })
                              }
                              placeholder={
                                filter.operator === "in" ||
                                filter.operator === "not in"
                                  ? '["en", "de"] or en,de'
                                  : 'en, true, 0.95, {"k":1}'
                              }
                            />
                          </label>

                          <button
                            type="button"
                            className="textButton filterRemoveButton"
                            onClick={() =>
                              removeStreamingFilter(entry.id, filter.id)
                            }
                          >
                            Remove
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                  <p className="fieldNote">
                    Values are detected automatically. For `in`/`not in`, use JSON or comma-separated values.
                  </p>
                </div>
              </div>
            </details>
          </div>
        ))}
      </div>
    </div>
  );
}
