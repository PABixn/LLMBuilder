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
        <summary>Core dataset settings</summary>
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
                Match the tokenizer trainer: choose one source mode and configure the full
                dataset stack here.
              </p>
            </div>

            <div className="sourceModeRow trainingTokenizerDatasetSection">
              <span>Dataset source</span>
              <div className="modeSwitch">
                <button
                  type="button"
                  className={`modeSwitchButton ${
                    datasetSourceMode === "local_file" ? "modeSwitchButton-active" : ""
                  }`}
                  onClick={selectLocalDatasetSource}
                >
                  Local files
                </button>
                <button
                  type="button"
                  className={`modeSwitchButton ${
                    datasetSourceMode === "streaming_hf" ? "modeSwitchButton-active" : ""
                  }`}
                  onClick={selectStreamingDatasetSource}
                >
                  Streaming Hugging Face datasets
                </button>
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
