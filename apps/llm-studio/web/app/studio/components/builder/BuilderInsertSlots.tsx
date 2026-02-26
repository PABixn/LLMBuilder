import type { DragEvent, MouseEvent } from "react";

import {
  labelForComponentKind,
  labelForMlpStepKind,
} from "../../utils/document";
import type {
  OpenInsertMenu,
  InsertMenuVariant,
} from "./types";

type OpenInsertMenuFromEvent = (
  event: MouseEvent<HTMLButtonElement>,
  config: Omit<OpenInsertMenu, "anchorEl">
) => void;

type CommonSlotProps = {
  openInsertMenu: OpenInsertMenu | null;
  openInsertMenuFromEvent: OpenInsertMenuFromEvent;
  dragOverKey: string | null;
};

type BlockInsertSlotProps = CommonSlotProps & {
  insertIndex: number;
  insertBlockAt: (insertIndex: number, preset: "default" | "empty") => void;
};

type ComponentInsertSlotProps = CommonSlotProps & {
  blockId: string;
  insertIndex: number;
  markDropTarget: (event: DragEvent<HTMLElement>, key: string) => void;
  handleDropComponent: (
    event: DragEvent<HTMLElement>,
    targetBlockId: string,
    insertIndex: number
  ) => void;
  insertComponentAt: (
    targetBlockId: string,
    insertIndex: number,
    componentKind: "attention" | "mlp" | "norm" | "activation"
  ) => void;
};

type MlpStepInsertSlotProps = CommonSlotProps & {
  blockId: string;
  componentId: string;
  insertIndex: number;
  markDropTarget: (event: DragEvent<HTMLElement>, key: string) => void;
  handleDropMlpStep: (
    event: DragEvent<HTMLElement>,
    targetBlockId: string,
    targetComponentId: string,
    insertIndex: number
  ) => void;
  insertMlpStepAt: (
    targetBlockId: string,
    targetComponentId: string,
    insertIndex: number,
    stepKind: "linear" | "norm" | "activation"
  ) => void;
};

function menuConfig(
  key: string,
  title: string,
  variant: InsertMenuVariant,
  items: OpenInsertMenu["items"]
) {
  return { key, title, variant, items };
}

export function BlockInsertSlot({
  insertIndex,
  insertBlockAt,
  openInsertMenu,
  openInsertMenuFromEvent,
}: BlockInsertSlotProps) {
  const menuKey = `block:${insertIndex}`;
  const isOpen = openInsertMenu?.key === menuKey;

  return (
    <div
      className={`dropSlot blockInsertSlot${isOpen ? " isOpen" : ""}`}
      data-insert-slot
    >
      <span className="dropSlotMark" aria-hidden />
      <button
        type="button"
        className="blockInsertTrigger"
        aria-label={`Add block at position ${insertIndex + 1}`}
        aria-haspopup="menu"
        aria-expanded={isOpen}
        title="Add block"
        onClick={(event) =>
          openInsertMenuFromEvent(
            event,
            menuConfig(menuKey, "Add block", "rail", [
              {
                id: "default",
                label: "Default transformer block",
                onSelect: () => insertBlockAt(insertIndex, "default"),
              },
              {
                id: "empty",
                label: "Empty block",
                onSelect: () => insertBlockAt(insertIndex, "empty"),
              },
            ])
          )
        }
      />
    </div>
  );
}

export function ComponentInsertSlot({
  blockId,
  insertIndex,
  openInsertMenu,
  openInsertMenuFromEvent,
  dragOverKey,
  markDropTarget,
  handleDropComponent,
  insertComponentAt,
}: ComponentInsertSlotProps) {
  const menuKey = `component:${blockId}:${insertIndex}`;
  const slotKey = `component-slot::${encodeURIComponent(blockId)}::${insertIndex}`;
  const isOpen = openInsertMenu?.key === menuKey;

  return (
    <div
      className={`dropSlot isCompact inlineInsertSlot${dragOverKey === slotKey ? " isActive" : ""}${isOpen ? " isOpen" : ""}`}
      data-insert-slot
      onDragOver={(event) => markDropTarget(event, slotKey)}
      onDrop={(event) => handleDropComponent(event, blockId, insertIndex)}
      aria-label="Insert component"
      title="Insert component"
    >
      <span className="dropSlotMark" aria-hidden />
      <button
        type="button"
        className="inlineInsertTrigger"
        aria-label={`Add component at position ${insertIndex + 1}`}
        aria-haspopup="menu"
        aria-expanded={isOpen}
        onDragOver={(event) => markDropTarget(event, slotKey)}
        onDrop={(event) => handleDropComponent(event, blockId, insertIndex)}
        onClick={(event) =>
          openInsertMenuFromEvent(
            event,
            menuConfig(
              menuKey,
              "Add component",
              "inline",
              (["attention", "mlp", "norm", "activation"] as const).map((kind) => ({
                id: kind,
                label: labelForComponentKind(kind),
                onSelect: () => insertComponentAt(blockId, insertIndex, kind),
              }))
            )
          )
        }
      />
    </div>
  );
}

export function MlpStepInsertSlot({
  blockId,
  componentId,
  insertIndex,
  openInsertMenu,
  openInsertMenuFromEvent,
  dragOverKey,
  markDropTarget,
  handleDropMlpStep,
  insertMlpStepAt,
}: MlpStepInsertSlotProps) {
  const menuKey = `mlp-step:${blockId}:${componentId}:${insertIndex}`;
  const slotKey = `mlp-slot::${encodeURIComponent(blockId)}::${encodeURIComponent(componentId)}::${insertIndex}`;
  const isOpen = openInsertMenu?.key === menuKey;

  return (
    <div
      className={`dropSlot inlineInsertSlot mlpStepInsertSlot${dragOverKey === slotKey ? " isActive" : ""}${isOpen ? " isOpen" : ""}`}
      data-insert-slot
      onDragOver={(event) => markDropTarget(event, slotKey)}
      onDrop={(event) => handleDropMlpStep(event, blockId, componentId, insertIndex)}
      aria-label="Insert MLP step"
      title="Insert MLP step"
    >
      <span className="dropSlotMark" aria-hidden />
      <button
        type="button"
        className="inlineInsertTrigger"
        aria-label={`Add MLP step at position ${insertIndex + 1}`}
        aria-haspopup="menu"
        aria-expanded={isOpen}
        onDragOver={(event) => markDropTarget(event, slotKey)}
        onDrop={(event) => handleDropMlpStep(event, blockId, componentId, insertIndex)}
        onClick={(event) =>
          openInsertMenuFromEvent(
            event,
            menuConfig(
              menuKey,
              "Add MLP step",
              "inline",
              (["linear", "norm", "activation"] as const).map((kind) => ({
                id: kind,
                label: labelForMlpStepKind(kind),
                onSelect: () => insertMlpStepAt(blockId, componentId, insertIndex, kind),
              }))
            )
          )
        }
      />
    </div>
  );
}
