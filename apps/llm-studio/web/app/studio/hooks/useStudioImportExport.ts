import { startTransition, type ChangeEvent, type Dispatch, type SetStateAction } from "react";

import type { ModelConfig } from "../../../lib/defaults";
import { downloadTextFile } from "../../../lib/downloads";

import type { StudioDocument } from "../types";
import { studioDocumentFromConfig, studioDocumentToConfig } from "../utils/document";
import { parseImportedModelConfig } from "../utils/validation";

type SetNoticeMessage = (tone: "info" | "success" | "error", message: string) => void;

type UseStudioImportExportArgs = {
  documentState: StudioDocument;
  setDocumentState: Dispatch<SetStateAction<StudioDocument>>;
  setImportDraft: Dispatch<SetStateAction<string>>;
  setNoticeMessage: SetNoticeMessage;
};

export interface StudioImportExportActions {
  applyImportText: (text: string) => boolean;
  importFromFile: (event: ChangeEvent<HTMLInputElement>) => Promise<boolean>;
  exportJson: () => void;
  copyJson: () => Promise<void>;
}

export function useStudioImportExport({
  documentState,
  setDocumentState,
  setImportDraft,
  setNoticeMessage,
}: UseStudioImportExportArgs): StudioImportExportActions {
  function applyImportText(text: string): boolean {
    try {
      const parsedJson = JSON.parse(text) as unknown;
      const imported = parseImportedModelConfig(parsedJson);
      if (!imported.config) {
        setNoticeMessage("error", `Import failed: ${imported.errors.slice(0, 3).join(" ")}`);
        return false;
      }
      startTransition(() => {
        setDocumentState(studioDocumentFromConfig(imported.config as ModelConfig));
      });
      setNoticeMessage("success", "Imported model JSON.");
      return true;
    } catch (error) {
      setNoticeMessage(
        "error",
        error instanceof Error ? `Import failed: ${error.message}` : "Import failed."
      );
      return false;
    }
  }

  async function importFromFile(event: ChangeEvent<HTMLInputElement>): Promise<boolean> {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) {
      return false;
    }
    try {
      const text = await file.text();
      setImportDraft(text);
      return applyImportText(text);
    } catch {
      setNoticeMessage("error", "Could not read the JSON file.");
      return false;
    }
  }

  function exportJson(): void {
    const modelConfig = studioDocumentToConfig(documentState);
    void downloadTextFile("model_config.json", JSON.stringify(modelConfig, null, 2))
      .then((result) => {
        if (result !== "cancelled") {
          setNoticeMessage("success", result === "native" ? "Saved model JSON." : "Downloaded model JSON.");
        }
      })
      .catch((error: unknown) => {
        setNoticeMessage(
          "error",
          error instanceof Error ? error.message : "Could not export model JSON."
        );
      });
  }

  async function copyJson(): Promise<void> {
    const modelConfig = studioDocumentToConfig(documentState);
    try {
      await navigator.clipboard.writeText(JSON.stringify(modelConfig, null, 2));
      setNoticeMessage("success", "Copied JSON.");
    } catch {
      setNoticeMessage("error", "Could not copy JSON.");
    }
  }

  return {
    applyImportText,
    importFromFile,
    exportJson,
    copyJson,
  };
}
