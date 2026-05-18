import { FiTrash2 } from "react-icons/fi";

import {
  FILTER_OPERATORS,
} from "../constants";
import type {
  FilterOperator,
  StreamingDatasetFormState,
  StreamingFilterFormState,
} from "../types";

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
        <span>HF token <small>optional</small></span>
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
        <button
          type="button"
          className="secondaryButton"
          onClick={addStreamingDataset}
        >
          Add dataset
        </button>
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
                <span>Dataset name</span>
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
                <span>Split</span>
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
                <span>Weight</span>
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
                <span>Text columns</span>
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
              <summary>Advanced options</summary>
              <div className="fieldGrid">
                <label className="fieldLabel">
                  <span>Dataset config <small>optional</small></span>
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
                    <span className="filterBuilderTitle">Filters (optional)</span>
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
                            <span>Column</span>
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
                            <span>Operator</span>
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
                            <span>Value</span>
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
