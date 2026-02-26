import { useDeferredValue, useRef } from "react";

import type { StudioPageViewProps } from "../components/StudioPageView";
import type { Diagnostic } from "../types";
import {
  collectBuilderMetrics,
  collectConsecutiveIdenticalBlockGroups,
  studioDocumentToConfig,
} from "../utils/document";
import { pushDiagnostic, validateLocalConfig } from "../utils/validation";
import {
  buildBackendDiagnostics,
  useStudioBackendAnalysis,
  useStudioBackendValidation,
} from "./useStudioBackend";
import {
  useStudioDocumentEditor,
} from "./useStudioDocumentEditor";
import { useStudioImportExport } from "./useStudioImportExport";
import { useStudioWorkspaceState } from "./useStudioWorkspaceState";

function buildValidationStatusLabel(totalErrors: number, totalWarnings: number): string {
  if (totalErrors > 0) {
    return `${totalErrors} error${totalErrors === 1 ? "" : "s"}`;
  }
  if (totalWarnings > 0) {
    return `${totalWarnings} warning${totalWarnings === 1 ? "" : "s"}`;
  }
  return "Clean";
}

export function useStudioPageController(): StudioPageViewProps {
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const workspace = useStudioWorkspaceState();
  const editor = useStudioDocumentEditor({
    documentState: workspace.documentState,
    setDocumentState: workspace.setDocumentState,
    setExpandedComponentIds: workspace.setExpandedComponentIds,
    setExpandedMlpStepIds: workspace.setExpandedMlpStepIds,
    setNoticeMessage: workspace.setNoticeMessage,
  });
  const io = useStudioImportExport({
    documentState: workspace.documentState,
    setDocumentState: workspace.setDocumentState,
    setImportDraft: workspace.setImportDraft,
    setNoticeMessage: workspace.setNoticeMessage,
  });

  const modelConfig = studioDocumentToConfig(workspace.documentState);
  const localDiagnostics = validateLocalConfig(modelConfig);
  const localErrors = localDiagnostics.filter((item) => item.level === "error");
  const localWarnings = localDiagnostics.filter((item) => item.level === "warning");
  const metrics = collectBuilderMetrics(workspace.documentState);
  const consecutiveBlockGroups = collectConsecutiveIdenticalBlockGroups(workspace.documentState.blocks);

  const expandedBlockGroupKeys = new Set(
    consecutiveBlockGroups
      .filter((group) => {
        if (group.count <= 1) {
          return false;
        }
        for (let index = group.startIndex; index <= group.endIndex; index += 1) {
          if (workspace.expandedRepeatedBlockIds.has(workspace.documentState.blocks[index].id)) {
            return true;
          }
        }
        return false;
      })
      .map((group) => group.key)
  );

  const compactJson = JSON.stringify(modelConfig);
  const deferredJsonSignature = useDeferredValue(compactJson);
  const previewJson = JSON.stringify(JSON.parse(deferredJsonSignature), null, 2);

  const backendValidation = useStudioBackendValidation({
    modelConfig,
    compactJson,
    localErrorCount: localErrors.length,
  });
  const { backendAnalysis, runBackendAnalysis } = useStudioBackendAnalysis({
    modelConfig,
    compactJson,
    localErrorCount: localErrors.length,
    setNoticeMessage: workspace.setNoticeMessage,
  });

  const backendDiagnostics = buildBackendDiagnostics(backendValidation);
  const diagnostics: Diagnostic[] = [...localDiagnostics, ...backendDiagnostics];
  const totalErrors = diagnostics.filter((item) => item.level === "error").length;
  const totalWarnings = diagnostics.filter((item) => item.level === "warning").length;
  const backendAnalysisStale =
    backendAnalysis.configSignature !== null && backendAnalysis.configSignature !== compactJson;
  const validationStatusLabel = buildValidationStatusLabel(totalErrors, totalWarnings);

  return {
    fileInputRef,
    importFromFile: io.importFromFile,
    theme: workspace.theme,
    setTheme: workspace.setTheme,
    addBlock: editor.addBlock,
    exportJson: io.exportJson,
    resetDefaults: editor.resetDefaults,
    notice: workspace.notice,
    validationStatusLabel,
    backendValidation,
    totalErrors,
    totalWarnings,
    metrics,
    lastSavedAt: workspace.lastSavedAt,
    documentState: workspace.documentState,
    updateBaseField: editor.updateBaseField,
    setDocumentState: workspace.setDocumentState,
    clearDragState: editor.clearDragState,
    diagnostics,
    localErrors,
    localWarnings,
    backendAnalysis,
    backendAnalysisStale,
    runBackendAnalysis,
    previewJson,
    copyJson: io.copyJson,
    importDraft: workspace.importDraft,
    setImportDraft: workspace.setImportDraft,
    applyImportText: io.applyImportText,
    modelConfig,
    consecutiveBlockGroups,
    dragOverKey: editor.dragOverKey,
    expandedComponentIds: workspace.expandedComponentIds,
    expandedMlpStepIds: workspace.expandedMlpStepIds,
    expandedBlockGroupKeys,
    expandAllCanvasNodes: workspace.expandAllCanvasNodes,
    collapseAllCanvasNodes: workspace.collapseAllCanvasNodes,
    toggleExpandedBlockGroup: workspace.toggleExpandedBlockGroup,
    toggleExpandedComponent: workspace.toggleExpandedComponent,
    toggleExpandedMlpStep: workspace.toggleExpandedMlpStep,
    duplicateBlock: editor.duplicateBlock,
    deleteBlock: editor.deleteBlock,
    removeComponent: editor.removeComponent,
    removeMlpStep: editor.removeMlpStep,
    insertComponentAt: editor.insertComponentAt,
    insertMlpStepAt: editor.insertMlpStepAt,
    updateComponent: editor.updateComponent,
    updateMlpStep: editor.updateMlpStep,
    insertBlockAt: editor.insertBlockAt,
    beginDragComponent: editor.beginDragComponent,
    beginDragMlpStep: editor.beginDragMlpStep,
    markDropTarget: editor.markDropTarget,
    handleDropComponent: editor.handleDropComponent,
    handleDropMlpStep: editor.handleDropMlpStep,
    handleDropAtHighlightedSlot: editor.handleDropAtHighlightedSlot,
  };
}
