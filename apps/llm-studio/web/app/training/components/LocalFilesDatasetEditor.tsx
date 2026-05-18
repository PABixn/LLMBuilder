import type {
  ChangeEventHandler,
  DragEventHandler,
} from "react";
import { FiX } from "react-icons/fi";

import { formatCharCount } from "../lib/files";
import type { LocalTrainFileFormState } from "../types";
import { HelpTooltip, InfoTooltip } from "../../shared/components/HelpTooltip";

interface LocalFilesDatasetEditorProps {
  clearLocalTrainFiles: () => void;
  handleLocalTrainFilesDragEnter: DragEventHandler<HTMLElement>;
  handleLocalTrainFilesDragLeave: DragEventHandler<HTMLElement>;
  handleLocalTrainFilesDragOver: DragEventHandler<HTMLElement>;
  handleLocalTrainFilesDrop: DragEventHandler<HTMLElement>;
  handleTrainFilesSelected: ChangeEventHandler<HTMLInputElement>;
  isDraggingTrainFiles: boolean;
  isUploadingTrainFile: boolean;
  localTrainFiles: LocalTrainFileFormState[];
  removeLocalTrainFile: (localFileId: string) => void;
}

export function LocalFilesDatasetEditor({
  clearLocalTrainFiles,
  handleLocalTrainFilesDragEnter,
  handleLocalTrainFilesDragLeave,
  handleLocalTrainFilesDragOver,
  handleLocalTrainFilesDrop,
  handleTrainFilesSelected,
  isDraggingTrainFiles,
  isUploadingTrainFile,
  localTrainFiles,
  removeLocalTrainFile,
}: LocalFilesDatasetEditorProps) {
  return (
    <div className="datasetConfigurator trainingTokenizerDatasetSection">
      <div
        className={`localFileManager ${
          isDraggingTrainFiles ? "localFileManager-dragging" : ""
        }`}
        onDragEnter={handleLocalTrainFilesDragEnter}
        onDragOver={handleLocalTrainFilesDragOver}
        onDragLeave={handleLocalTrainFilesDragLeave}
        onDrop={handleLocalTrainFilesDrop}
      >
        <div className="localFileManagerHeader">
          <div>
            <strong>
              Local training files
              <InfoTooltip label="Local training files explanation" align="left" width="wide">
                <p>
                  Files are uploaded to the backend and referenced by the dataloader. Training
                  and evaluation read text from these files; duplicate file paths are skipped.
                </p>
              </InfoTooltip>
            </strong>
            <p>Training and evaluation use these files.</p>
          </div>
          <div className="localFileHeaderActions">
            <div className="localFileHeaderButtons">
              <HelpTooltip label="Add local training files" content="Select one or more text files. The app uploads them, records file size and character counts when available, and includes them in the next preflight check.">
                <label
                  className={`secondaryButton localFileUploadButton localFileHeaderButton ${
                    isUploadingTrainFile ? "localFileUploadButton-disabled" : ""
                  }`}
                  aria-disabled={isUploadingTrainFile}
                >
                  {isUploadingTrainFile ? "Uploading..." : "Add files"}
                  <input
                    type="file"
                    multiple
                    onChange={handleTrainFilesSelected}
                    disabled={isUploadingTrainFile}
                  />
                </label>
              </HelpTooltip>
              <HelpTooltip label="Remove all local files" content="Clears the file list from this configuration. It does not delete original files from your computer.">
                <button
                  type="button"
                  className="textButton localFileHeaderButton"
                  onClick={clearLocalTrainFiles}
                  disabled={localTrainFiles.length === 0}
                >
                  Remove all
                </button>
              </HelpTooltip>
            </div>
            <span className="localFileCount">
              {localTrainFiles.length} file
              {localTrainFiles.length === 1 ? "" : "s"}
            </span>
          </div>
        </div>

        {localTrainFiles.length === 0 ? (
          <p className="filterEmpty">
            No files added yet.
          </p>
        ) : (
          <ul className="localFileList">
            {localTrainFiles.map((entry) => {
              const fileCharLabel = formatCharCount(entry.sizeChars);
              return (
                <li key={entry.id} className="localFileItem">
                  <strong className="localFileName" title={entry.fileName}>
                    {entry.fileName}
                  </strong>
                  <div className="localFileActions">
                    <span className="localFileStat">
                      {fileCharLabel
                        ? `${fileCharLabel} chars`
                        : "Char count pending"}
                    </span>
                    <button
                      type="button"
                      className="textButton localFileRemoveIconButton"
                      onClick={() => removeLocalTrainFile(entry.id)}
                      aria-label={`Remove ${entry.fileName}`}
                      title={`Remove ${entry.fileName}`}
                    >
                      <FiX aria-hidden="true" />
                    </button>
                  </div>
                </li>
              );
            })}
          </ul>
        )}

        <span className="fieldNote">Duplicate file paths are ignored.</span>
      </div>
    </div>
  );
}
