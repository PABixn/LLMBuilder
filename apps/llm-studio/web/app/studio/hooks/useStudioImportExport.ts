import { startTransition, type ChangeEvent, type Dispatch, type SetStateAction } from "react";

import type { ModelConfig } from "../../../lib/defaults";

import type { StudioDocument } from "../types";
import { studioDocumentFromConfig, studioDocumentToConfig } from "../utils/document";
import { downloadTextFile } from "../utils/format";
import { parseImportedModelConfig } from "../utils/validation";

type SetNoticeMessage = (tone: "info" | "success" | "error", message: string) => void;

type UseStudioImportExportArgs = {
  documentState: StudioDocument;
  setDocumentState: Dispatch<SetStateAction<StudioDocument>>;
  setImportDraft: Dispatch<SetStateAction<string>>;
  setNoticeMessage: SetNoticeMessage;
};

export interface StudioImportExportActions {
  applyImportText: (text: string) => void;
  importFromFile: (event: ChangeEvent<HTMLInputElement>) => Promise<void>;
  exportJson: () => void;
  copyJson: () => Promise<void>;
}

export function useStudioImportExport({
  documentState,
  setDocumentState,
  setImportDraft,
  setNoticeMessage,
}: UseStudioImportExportArgs): StudioImportExportActions {
  function applyImportText(text: string): void {
    try {
      const parsedJson = JSON.parse(text) as unknown;
      const imported = parseImportedModelConfig(parsedJson);
      if (!imported.config) {
        setNoticeMessage("error", `Import failed: ${imported.errors.slice(0, 3).join(" ")}`);
        return;
      }
      startTransition(() => {
        setDocumentState(studioDocumentFromConfig(imported.config as ModelConfig));
      });
      setNoticeMessage("success", "Imported model config JSON into visual builder.");
    } catch (error) {
      setNoticeMessage(
        "error",
        error instanceof Error ? `Import failed: ${error.message}` : "Import failed."
      );
    }
  }

  async function importFromFile(event: ChangeEvent<HTMLInputElement>): Promise<void> {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) {
      return;
    }
    try {
      const text = await file.text();
      setImportDraft(text);
      applyImportText(text);
    } catch {
      setNoticeMessage("error", "Failed to read selected JSON file.");
    }
  }

  function exportJson(): void {
    const modelConfig = studioDocumentToConfig(documentState);
    downloadTextFile("model_config.json", JSON.stringify(modelConfig, null, 2));
    setNoticeMessage("success", "Exported model config JSON.");
  }

  async function copyJson(): Promise<void> {
    const modelConfig = studioDocumentToConfig(documentState);
    try {
      await navigator.clipboard.writeText(JSON.stringify(modelConfig, null, 2));
      setNoticeMessage("success", "Copied JSON to clipboard.");
    } catch {
      setNoticeMessage("error", "Clipboard write failed in this environment.");
    }
  }

  return {
    applyImportText,
    importFromFile,
    exportJson,
    copyJson,
  };
}
