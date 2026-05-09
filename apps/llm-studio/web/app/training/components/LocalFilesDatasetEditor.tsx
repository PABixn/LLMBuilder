import type {
  ChangeEventHandler,
  DragEventHandler,
} from "react";
import { FiX } from "react-icons/fi";

import { formatCharCount } from "../lib/files";
import type { LocalTrainFileFormState } from "../types";

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
            <strong>Local training files</strong>
            <p>Training and evaluation use this same file set.</p>
          </div>
          <div className="localFileHeaderActions">
            <div className="localFileHeaderButtons">
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
              <button
                type="button"
                className="textButton localFileHeaderButton"
                onClick={clearLocalTrainFiles}
                disabled={localTrainFiles.length === 0}
              >
                Remove all
              </button>
            </div>
            <span className="localFileCount">
              {localTrainFiles.length} file
              {localTrainFiles.length === 1 ? "" : "s"}
            </span>
          </div>
        </div>

        {localTrainFiles.length === 0 ? (
          <p className="filterEmpty">
            No local files added yet. Add one or more files to train.
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
                        ? `${fileCharLabel} characters`
                        : "Character count pending"}
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

        <span className="fieldNote">Files are deduplicated by stored path.</span>
      </div>
    </div>
  );
}
