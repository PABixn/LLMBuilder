import type { DragEvent } from "react";

import type {
  BlockInsertPreset,
  BuilderMetrics,
  ConsecutiveBlockGroup,
  MlpStepKind,
  StudioComponent,
  StudioComponentKind,
  StudioDocument,
  StudioMlpStep,
} from "../../types";

export interface BuilderPanelProps {
  documentState: StudioDocument;
  metrics: BuilderMetrics;
  consecutiveBlockGroups: ConsecutiveBlockGroup[];
  dragOverKey: string | null;
  expandedComponentIds: Set<string>;
  expandedMlpStepIds: Set<string>;
  expandedBlockGroupKeys: Set<string>;
  addBlock: () => void;
  expandAllCanvasNodes: () => void;
  collapseAllCanvasNodes: () => void;
  toggleExpandedBlockGroup: (groupKey: string) => void;
  toggleExpandedComponent: (componentId: string) => void;
  toggleExpandedMlpStep: (stepId: string) => void;
  duplicateBlock: (blockId: string) => void;
  deleteBlock: (blockId: string) => void;
  removeComponent: (blockId: string, componentId: string) => void;
  removeMlpStep: (blockId: string, componentId: string, stepId: string) => void;
  insertComponentAt: (
    targetBlockId: string,
    insertIndex: number,
    componentKind: StudioComponentKind
  ) => void;
  insertMlpStepAt: (
    targetBlockId: string,
    targetComponentId: string,
    insertIndex: number,
    stepKind: MlpStepKind
  ) => void;
  updateComponent: (
    blockId: string,
    componentId: string,
    updater: (component: StudioComponent) => StudioComponent
  ) => void;
  updateMlpStep: (
    blockId: string,
    componentId: string,
    stepId: string,
    updater: (step: StudioMlpStep) => StudioMlpStep
  ) => void;
  insertBlockAt: (insertIndex: number, preset: BlockInsertPreset) => void;
  beginDragComponent: (
    event: DragEvent<HTMLDivElement>,
    fromBlockId: string,
    componentId: string
  ) => void;
  beginDragMlpStep: (
    event: DragEvent<HTMLDivElement>,
    fromBlockId: string,
    fromComponentId: string,
    stepId: string
  ) => void;
  clearDragState: () => void;
  markDropTarget: (event: DragEvent<HTMLElement>, key: string) => void;
  handleDropComponent: (
    event: DragEvent<HTMLElement>,
    targetBlockId: string,
    insertIndex: number
  ) => void;
  handleDropMlpStep: (
    event: DragEvent<HTMLElement>,
    targetBlockId: string,
    targetComponentId: string,
    insertIndex: number
  ) => void;
  handleDropAtHighlightedSlot: (event: DragEvent<HTMLElement>) => void;
}

export type InsertMenuVariant = "rail" | "inline";

export type InsertMenuAction = {
  id: string;
  label: string;
  onSelect: () => void;
};

export type OpenInsertMenu = {
  key: string;
  title: string;
  variant: InsertMenuVariant;
  anchorEl: HTMLButtonElement;
  items: InsertMenuAction[];
};
