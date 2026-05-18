import {
  forwardRef,
  type ChangeEventHandler,
  type DragEventHandler,
  type Ref,
} from "react";

import type {
  DatasetSourceMode,
  LocalTrainFileFormState,
  StreamingDatasetFormState,
  StreamingFilterFormState,
} from "../types";
import { LocalFilesDatasetEditor } from "./LocalFilesDatasetEditor";
import { StreamingDatasetEditor } from "./StreamingDatasetEditor";
import { HelpTooltip, InfoTooltip } from "../../shared/components/HelpTooltip";

interface DatasetSettingsPanelProps {
  addStreamingDataset: () => void;
  addStreamingFilter: (datasetId: string) => void;
  clearLocalTrainFiles: () => void;
  datasetSettingsRef: Ref<HTMLDivElement>;
  datasetSourceMode: DatasetSourceMode;
  handleLoadStreamingTemplate: () => void;
  handleLocalTrainFilesDragEnter: DragEventHandler<HTMLElement>;
  handleLocalTrainFilesDragLeave: DragEventHandler<HTMLElement>;
  handleLocalTrainFilesDragOver: DragEventHandler<HTMLElement>;
  handleLocalTrainFilesDrop: DragEventHandler<HTMLElement>;
  handleTrainFilesSelected: ChangeEventHandler<HTMLInputElement>;
  hfToken: string;
  highlighted: boolean;
  isDraggingTrainFiles: boolean;
  isLoadingDatasetTemplate: boolean;
  isUploadingTrainFile: boolean;
  localTrainFiles: LocalTrainFileFormState[];
  removeLocalTrainFile: (localFileId: string) => void;
  removeStreamingDataset: (datasetId: string) => void;
  removeStreamingFilter: (datasetId: string, filterId: string) => void;
  selectLocalDatasetSource: () => void;
  selectStreamingDatasetSource: () => void;
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

export const DatasetSettingsPanel = forwardRef<HTMLDetailsElement, DatasetSettingsPanelProps>(
  function DatasetSettingsPanel(
    {
      addStreamingDataset,
      addStreamingFilter,
      clearLocalTrainFiles,
      datasetSettingsRef,
      datasetSourceMode,
      handleLoadStreamingTemplate,
      handleLocalTrainFilesDragEnter,
      handleLocalTrainFilesDragLeave,
      handleLocalTrainFilesDragOver,
      handleLocalTrainFilesDrop,
      handleTrainFilesSelected,
      hfToken,
      highlighted,
      isDraggingTrainFiles,
      isLoadingDatasetTemplate,
      isUploadingTrainFile,
      localTrainFiles,
      removeLocalTrainFile,
      removeStreamingDataset,
      removeStreamingFilter,
      selectLocalDatasetSource,
      selectStreamingDatasetSource,
      setHfToken,
      streamingDatasets,
      updateStreamingDataset,
      updateStreamingFilter,
      updateStreamingWeight,
    },
    ref
  ) {
    return (
      <details className="settingsPanel" open ref={ref}>
        <summary>
          <span>Dataset settings</span>
          <InfoTooltip label="Dataset settings explanation" align="right" width="wide">
            <strong>Dataset settings</strong>
            <p>
              Pick the text source for training. Local files are uploaded into this
              workspace; streaming datasets are read from Hugging Face when the run starts.
            </p>
          </InfoTooltip>
        </summary>
        <div className="settingsGrid">
          <div
            id="settings-dataset"
            ref={datasetSettingsRef}
            className={`settingsGroup settingsCategoryAnchor ${
              highlighted ? "settingsCategoryAnchor-highlight" : ""
            }`}
          >
            <div className="settingsGroupHeader">
              <h3>Dataset sources</h3>
              <p className="settingsGroupHint">
                Choose local files or streaming datasets.
              </p>
            </div>

            <div className="sourceModeRow trainingTokenizerDatasetSection">
              <span className="fieldLabelText">
                <span>Dataset source</span>
                <InfoTooltip label="Dataset source explanation" width="wide">
                  <strong>Dataset source</strong>
                  <p>
                    Use local files for small or private text you upload here. Use streaming
                    datasets for large Hugging Face datasets that should be sampled during training.
                  </p>
                </InfoTooltip>
              </span>
              <div className="modeSwitch">
                <HelpTooltip label="Local files dataset mode" content="Train from text files uploaded into this app. The app stores the uploaded file references in the dataloader config and ignores duplicate paths.">
                  <button
                    type="button"
                    className={`modeSwitchButton ${
                      datasetSourceMode === "local_file" ? "modeSwitchButton-active" : ""
                    }`}
                    onClick={selectLocalDatasetSource}
                  >
                    Local files
                  </button>
                </HelpTooltip>
                <HelpTooltip label="Streaming datasets mode" content="Train from Hugging Face datasets without uploading the whole dataset first. Weights, splits, text columns, and optional filters decide what text enters training.">
                  <button
                    type="button"
                    className={`modeSwitchButton ${
                      datasetSourceMode === "streaming_hf" ? "modeSwitchButton-active" : ""
                    }`}
                    onClick={selectStreamingDatasetSource}
                  >
                    Streaming datasets
                  </button>
                </HelpTooltip>
              </div>
            </div>

            {datasetSourceMode === "local_file" ? (
              <LocalFilesDatasetEditor
                clearLocalTrainFiles={clearLocalTrainFiles}
                handleLocalTrainFilesDragEnter={handleLocalTrainFilesDragEnter}
                handleLocalTrainFilesDragLeave={handleLocalTrainFilesDragLeave}
                handleLocalTrainFilesDragOver={handleLocalTrainFilesDragOver}
                handleLocalTrainFilesDrop={handleLocalTrainFilesDrop}
                handleTrainFilesSelected={handleTrainFilesSelected}
                isDraggingTrainFiles={isDraggingTrainFiles}
                isUploadingTrainFile={isUploadingTrainFile}
                localTrainFiles={localTrainFiles}
                removeLocalTrainFile={removeLocalTrainFile}
              />
            ) : (
              <StreamingDatasetEditor
                addStreamingDataset={addStreamingDataset}
                addStreamingFilter={addStreamingFilter}
                handleLoadStreamingTemplate={handleLoadStreamingTemplate}
                hfToken={hfToken}
                isLoadingDatasetTemplate={isLoadingDatasetTemplate}
                removeStreamingDataset={removeStreamingDataset}
                removeStreamingFilter={removeStreamingFilter}
                setHfToken={setHfToken}
                streamingDatasets={streamingDatasets}
                updateStreamingDataset={updateStreamingDataset}
                updateStreamingFilter={updateStreamingFilter}
                updateStreamingWeight={updateStreamingWeight}
              />
            )}
          </div>
        </div>
      </details>
    );
  }
);
