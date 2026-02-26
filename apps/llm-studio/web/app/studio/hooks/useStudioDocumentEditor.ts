import { useRef, useState, type Dispatch, type DragEvent, type SetStateAction } from "react";

import { createDefaultBlockConfig, createDefaultModelConfig } from "../../../lib/defaults";

import type {
  BlockInsertPreset,
  DragPayload,
  MlpStepKind,
  StudioComponent,
  StudioComponentKind,
  StudioDocument,
  StudioDocumentNumericField,
  StudioMlpStep,
} from "../types";
import { DND_MIME } from "../types";
import {
  clamp,
  clone,
  cloneBlockWithNewIds,
  createDefaultStudioComponent,
  createDefaultStudioMlpStep,
  findBlockIndex,
  findComponentIndex,
  getMlpComponent,
  studioBlockFromConfig,
  studioDocumentFromConfig,
} from "../utils/document";

type SetIdSet = Dispatch<SetStateAction<Set<string>>>;

type SetNoticeMessage = (tone: "info" | "success" | "error", message: string) => void;

export interface StudioDocumentEditor {
  dragOverKey: string | null;
  updateBaseField: (key: StudioDocumentNumericField, value: number) => void;
  insertBlockAt: (insertIndex: number, preset: BlockInsertPreset) => void;
  addBlock: () => void;
  duplicateBlock: (blockId: string) => void;
  deleteBlock: (blockId: string) => void;
  resetDefaults: () => void;
  removeComponent: (blockId: string, componentId: string) => void;
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
  clearDragState: () => void;
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

type UseStudioDocumentEditorArgs = {
  documentState: StudioDocument;
  setDocumentState: Dispatch<SetStateAction<StudioDocument>>;
  setExpandedComponentIds: SetIdSet;
  setExpandedMlpStepIds: SetIdSet;
  setNoticeMessage: SetNoticeMessage;
};

export function useStudioDocumentEditor({
  documentState,
  setDocumentState,
  setExpandedComponentIds,
  setExpandedMlpStepIds,
  setNoticeMessage,
}: UseStudioDocumentEditorArgs): StudioDocumentEditor {
  const dragRef = useRef<DragPayload | null>(null);
  const [dragOverKey, setDragOverKey] = useState<string | null>(null);

  function updateBaseField(key: StudioDocumentNumericField, value: number): void {
    setDocumentState((current) => ({ ...current, [key]: value }));
  }

  function insertBlockAt(insertIndex: number, preset: BlockInsertPreset): void {
    setDocumentState((current) => {
      const nextBlocks = current.blocks.slice();
      const nextBlock =
        preset === "empty"
          ? studioBlockFromConfig({ components: [] })
          : studioBlockFromConfig(createDefaultBlockConfig());
      nextBlocks.splice(clamp(insertIndex, 0, nextBlocks.length), 0, nextBlock);
      return { ...current, blocks: nextBlocks };
    });
  }

  function addBlock(): void {
    insertBlockAt(Number.MAX_SAFE_INTEGER, "default");
  }

  function duplicateBlock(blockId: string): void {
    setDocumentState((current) => {
      const index = current.blocks.findIndex((block) => block.id === blockId);
      if (index < 0) {
        return current;
      }
      const duplicate = cloneBlockWithNewIds(current.blocks[index]);
      const nextBlocks = current.blocks.slice();
      nextBlocks.splice(index + 1, 0, duplicate);
      return { ...current, blocks: nextBlocks };
    });
  }

  function deleteBlock(blockId: string): void {
    setDocumentState((current) => {
      if (current.blocks.length <= 1) {
        return current;
      }
      return {
        ...current,
        blocks: current.blocks.filter((block) => block.id !== blockId),
      };
    });
  }

  function resetDefaults(): void {
    setDocumentState(studioDocumentFromConfig(createDefaultModelConfig()));
    setNoticeMessage("info", "Reset to default LLM config template.");
  }

  function removeComponent(blockId: string, componentId: string): void {
    setDocumentState((current) => {
      const next = clone(current);
      const blockIndex = findBlockIndex(next, blockId);
      if (blockIndex < 0) {
        return current;
      }
      next.blocks[blockIndex].components = next.blocks[blockIndex].components.filter(
        (component) => component.id !== componentId
      );
      return next;
    });
  }

  function updateComponent(
    blockId: string,
    componentId: string,
    updater: (component: StudioComponent) => StudioComponent
  ): void {
    setDocumentState((current) => {
      const next = clone(current);
      const blockIndex = findBlockIndex(next, blockId);
      if (blockIndex < 0) {
        return current;
      }
      const componentIndex = findComponentIndex(next.blocks[blockIndex], componentId);
      if (componentIndex < 0) {
        return current;
      }
      next.blocks[blockIndex].components[componentIndex] = updater(
        next.blocks[blockIndex].components[componentIndex]
      );
      return next;
    });
  }

  function updateMlpStep(
    blockId: string,
    componentId: string,
    stepId: string,
    updater: (step: StudioMlpStep) => StudioMlpStep
  ): void {
    setDocumentState((current) => {
      const next = clone(current);
      const mlpRef = getMlpComponent(next, blockId, componentId);
      if (!mlpRef) {
        return current;
      }
      const stepIndex = mlpRef.component.mlp.sequence.findIndex((step) => step.id === stepId);
      if (stepIndex < 0) {
        return current;
      }
      mlpRef.component.mlp.sequence[stepIndex] = updater(mlpRef.component.mlp.sequence[stepIndex]);
      return next;
    });
  }

  function removeMlpStep(blockId: string, componentId: string, stepId: string): void {
    setDocumentState((current) => {
      const next = clone(current);
      const mlpRef = getMlpComponent(next, blockId, componentId);
      if (!mlpRef) {
        return current;
      }
      mlpRef.component.mlp.sequence = mlpRef.component.mlp.sequence.filter(
        (step) => step.id !== stepId
      );
      return next;
    });
  }

  function insertComponentAt(
    targetBlockId: string,
    insertIndex: number,
    componentKind: StudioComponentKind
  ): void {
    const createdComponent = createDefaultStudioComponent(componentKind);

    setDocumentState((current) => {
      const next = clone(current);
      const targetBlockIndex = findBlockIndex(next, targetBlockId);
      if (targetBlockIndex < 0) {
        return current;
      }
      const targetBlock = next.blocks[targetBlockIndex];
      const targetInsertIndex = clamp(insertIndex, 0, targetBlock.components.length);
      targetBlock.components.splice(targetInsertIndex, 0, createdComponent);
      return next;
    });

    setExpandedComponentIds((current) => new Set([...current, createdComponent.id]));
    if (createdComponent.kind === "mlp") {
      setExpandedMlpStepIds(
        (current) => new Set([...current, ...createdComponent.mlp.sequence.map((step) => step.id)])
      );
    }
  }

  function insertMlpStepAt(
    targetBlockId: string,
    targetComponentId: string,
    insertIndex: number,
    stepKind: MlpStepKind
  ): void {
    const createdStep = createDefaultStudioMlpStep(stepKind);

    setDocumentState((current) => {
      const next = clone(current);
      const targetRef = getMlpComponent(next, targetBlockId, targetComponentId);
      if (!targetRef) {
        return current;
      }
      const targetSequence = targetRef.component.mlp.sequence;
      const targetInsertIndex = clamp(insertIndex, 0, targetSequence.length);
      targetSequence.splice(targetInsertIndex, 0, createdStep);
      return next;
    });

    setExpandedMlpStepIds((current) => new Set([...current, createdStep.id]));
  }

  function writeDragPayload(event: DragEvent, payload: DragPayload): void {
    const serializedPayload = JSON.stringify(payload);
    dragRef.current = payload;
    event.dataTransfer.setData(DND_MIME, serializedPayload);
    event.dataTransfer.setData("text/plain", serializedPayload);
    event.dataTransfer.effectAllowed = "move";
  }

  function readDragPayload(event: DragEvent): DragPayload | null {
    if (dragRef.current) {
      return dragRef.current;
    }
    const raw =
      event.dataTransfer.getData(DND_MIME) || event.dataTransfer.getData("text/plain");
    if (!raw) {
      return null;
    }
    try {
      return JSON.parse(raw) as DragPayload;
    } catch {
      return null;
    }
  }

  function clearDragState(): void {
    setDragOverKey(null);
    // `dragend` can fire before `drop` in some browsers; defer clearing the payload
    // so the drop handler can still read it during the same turn.
    window.setTimeout(() => {
      dragRef.current = null;
    }, 0);
  }

  function beginDragComponent(
    event: DragEvent<HTMLDivElement>,
    fromBlockId: string,
    componentId: string
  ): void {
    event.stopPropagation();
    writeDragPayload(event, { kind: "block-component", fromBlockId, componentId });
  }

  function beginDragMlpStep(
    event: DragEvent<HTMLDivElement>,
    fromBlockId: string,
    fromComponentId: string,
    stepId: string
  ): void {
    event.stopPropagation();
    writeDragPayload(event, { kind: "mlp-step", fromBlockId, fromComponentId, stepId });
  }

  function markDropTarget(event: DragEvent<HTMLElement>, key: string): void {
    event.preventDefault();
    setDragOverKey(key);
    event.dataTransfer.dropEffect = "move";
  }

  function handleDropComponent(
    event: DragEvent<HTMLElement>,
    targetBlockId: string,
    insertIndex: number
  ): void {
    event.preventDefault();
    const payload = readDragPayload(event);
    clearDragState();
    if (!payload) {
      return;
    }

    if (payload.kind === "palette-component") {
      insertComponentAt(targetBlockId, insertIndex, payload.componentKind);
      return;
    }

    setDocumentState((current) => {
      const next = clone(current);
      const targetBlockIndex = findBlockIndex(next, targetBlockId);
      if (targetBlockIndex < 0) {
        return current;
      }
      const targetBlock = next.blocks[targetBlockIndex];
      const targetInsertIndex = clamp(insertIndex, 0, targetBlock.components.length);

      if (payload.kind !== "block-component") {
        return current;
      }

      const sourceBlockIndex = findBlockIndex(next, payload.fromBlockId);
      if (sourceBlockIndex < 0) {
        return current;
      }
      const sourceBlock = next.blocks[sourceBlockIndex];
      const sourceComponentIndex = findComponentIndex(sourceBlock, payload.componentId);
      if (sourceComponentIndex < 0) {
        return current;
      }
      const [moved] = sourceBlock.components.splice(sourceComponentIndex, 1);
      let adjustedInsertIndex = targetInsertIndex;
      if (payload.fromBlockId === targetBlockId && sourceComponentIndex < adjustedInsertIndex) {
        adjustedInsertIndex -= 1;
      }
      adjustedInsertIndex = clamp(adjustedInsertIndex, 0, targetBlock.components.length);
      targetBlock.components.splice(adjustedInsertIndex, 0, moved);
      return next;
    });
  }

  function handleDropMlpStep(
    event: DragEvent<HTMLElement>,
    targetBlockId: string,
    targetComponentId: string,
    insertIndex: number
  ): void {
    event.preventDefault();
    const payload = readDragPayload(event);
    clearDragState();
    if (!payload) {
      return;
    }

    if (payload.kind === "palette-mlp-step") {
      insertMlpStepAt(targetBlockId, targetComponentId, insertIndex, payload.stepKind);
      return;
    }

    setDocumentState((current) => {
      const next = clone(current);
      const targetRef = getMlpComponent(next, targetBlockId, targetComponentId);
      if (!targetRef) {
        return current;
      }
      const targetSequence = targetRef.component.mlp.sequence;
      let targetInsertIndex = clamp(insertIndex, 0, targetSequence.length);

      if (payload.kind !== "mlp-step") {
        return current;
      }

      const sourceRef = getMlpComponent(next, payload.fromBlockId, payload.fromComponentId);
      if (!sourceRef) {
        return current;
      }
      const sourceIndex = sourceRef.component.mlp.sequence.findIndex(
        (step) => step.id === payload.stepId
      );
      if (sourceIndex < 0) {
        return current;
      }

      const [moved] = sourceRef.component.mlp.sequence.splice(sourceIndex, 1);
      if (
        payload.fromBlockId === targetBlockId &&
        payload.fromComponentId === targetComponentId &&
        sourceIndex < targetInsertIndex
      ) {
        targetInsertIndex -= 1;
      }
      targetInsertIndex = clamp(targetInsertIndex, 0, targetRef.component.mlp.sequence.length);
      targetRef.component.mlp.sequence.splice(targetInsertIndex, 0, moved);
      return next;
    });
  }

  function handleDropAtHighlightedSlot(event: DragEvent<HTMLElement>): void {
    const eventTarget = event.target;
    if (eventTarget instanceof Element && eventTarget.closest("[data-insert-slot]")) {
      return;
    }
    if (!dragOverKey) {
      return;
    }

    const parts = dragOverKey.split("::");
    if (parts.length === 3 && parts[0] === "component-slot") {
      const [, encodedBlockId, rawInsertIndex] = parts;
      const insertIndex = Number.parseInt(rawInsertIndex, 10);
      if (!Number.isFinite(insertIndex)) {
        return;
      }
      handleDropComponent(event, decodeURIComponent(encodedBlockId), insertIndex);
      return;
    }

    if (parts.length === 4 && parts[0] === "mlp-slot") {
      const [, encodedBlockId, encodedComponentId, rawInsertIndex] = parts;
      const insertIndex = Number.parseInt(rawInsertIndex, 10);
      if (!Number.isFinite(insertIndex)) {
        return;
      }
      handleDropMlpStep(
        event,
        decodeURIComponent(encodedBlockId),
        decodeURIComponent(encodedComponentId),
        insertIndex
      );
    }
  }

  return {
    dragOverKey,
    updateBaseField,
    insertBlockAt,
    addBlock,
    duplicateBlock,
    deleteBlock,
    resetDefaults,
    removeComponent,
    updateComponent,
    updateMlpStep,
    removeMlpStep,
    insertComponentAt,
    insertMlpStepAt,
    clearDragState,
    beginDragComponent,
    beginDragMlpStep,
    markDropTarget,
    handleDropComponent,
    handleDropMlpStep,
    handleDropAtHighlightedSlot,
  };
}
