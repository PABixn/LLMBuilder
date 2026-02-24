import type { DragEvent, KeyboardEvent as ReactKeyboardEvent, MouseEvent, ReactNode } from "react";
import { Fragment, useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  FiChevronDown,
  FiChevronRight,
  FiCopy,
  FiLayers,
  FiMove,
  FiPlus,
  FiTrash2,
} from "react-icons/fi";

import { ACTIVATION_TYPES, type ActivationType, type NormConfig } from "../../../../lib/defaults";

import type {
  BlockInsertPreset,
  BuilderMetrics,
  ConsecutiveBlockGroup,
  MlpStepKind,
  StudioBlock,
  StudioComponent,
  StudioComponentKind,
  StudioDocument,
  StudioMlpStep,
} from "../../types";
import {
  labelForComponentKind,
  labelForMlpStepKind,
  summarizeComponent,
  summarizeMlpStep,
} from "../../utils/document";
import {
  componentDomIdPrefix,
  integerInputValue,
  mlpStepDomIdPrefix,
  numberInputValue,
  parseIntegerInput,
  parseNumberInput,
} from "../../utils/format";

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
  markDropTarget: (event: DragEvent<HTMLDivElement>, key: string) => void;
  handleDropComponent: (
    event: DragEvent<HTMLDivElement>,
    targetBlockId: string,
    insertIndex: number
  ) => void;
  handleDropMlpStep: (
    event: DragEvent<HTMLDivElement>,
    targetBlockId: string,
    targetComponentId: string,
    insertIndex: number
  ) => void;
}

type InsertMenuVariant = "rail" | "inline";

type InsertMenuAction = {
  id: string;
  label: string;
  onSelect: () => void;
};

type OpenInsertMenu = {
  key: string;
  title: string;
  variant: InsertMenuVariant;
  anchorEl: HTMLButtonElement;
  items: InsertMenuAction[];
};

export function BuilderPanel({
  documentState,
  metrics,
  consecutiveBlockGroups,
  dragOverKey,
  expandedComponentIds,
  expandedMlpStepIds,
  expandedBlockGroupKeys,
  addBlock,
  expandAllCanvasNodes,
  collapseAllCanvasNodes,
  toggleExpandedBlockGroup,
  toggleExpandedComponent,
  toggleExpandedMlpStep,
  duplicateBlock,
  deleteBlock,
  removeComponent,
  removeMlpStep,
  insertComponentAt,
  insertMlpStepAt,
  updateComponent,
  updateMlpStep,
  insertBlockAt,
  beginDragComponent,
  beginDragMlpStep,
  clearDragState,
  markDropTarget,
  handleDropComponent,
  handleDropMlpStep,
}: BuilderPanelProps) {
  const [openInsertMenu, setOpenInsertMenu] = useState<OpenInsertMenu | null>(null);
  const [insertMenuPosition, setInsertMenuPosition] = useState<{ left: number; top: number } | null>(
    null
  );
  const insertMenuRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (openInsertMenu === null) {
      return;
    }

    function handlePointerDown(event: PointerEvent): void {
      const target = event.target;
      if (target instanceof Element && target.closest("[data-insert-slot]")) {
        return;
      }
      closeInsertMenu();
    }

    function handleKeyDown(event: KeyboardEvent): void {
      if (event.key === "Escape") {
        closeInsertMenu();
      }
    }

    window.addEventListener("pointerdown", handlePointerDown);
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("pointerdown", handlePointerDown);
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [openInsertMenu]);

  useLayoutEffect(() => {
    if (!openInsertMenu || typeof window === "undefined") {
      setInsertMenuPosition(null);
      return;
    }

    let rafId = 0;

    const updatePosition = () => {
      const anchorEl = openInsertMenu.anchorEl;
      if (!anchorEl.isConnected || !document.body.contains(anchorEl)) {
        setOpenInsertMenu(null);
        setInsertMenuPosition(null);
        return;
      }

      const anchorRect = anchorEl.getBoundingClientRect();
      const menuRect = insertMenuRef.current?.getBoundingClientRect();
      const menuWidth = menuRect?.width ?? (openInsertMenu.variant === "rail" ? 204 : 156);
      const menuHeight = menuRect?.height ?? 180;

      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;
      const gap = 8;
      const margin = 8;

      let left = 0;
      let top = 0;

      if (openInsertMenu.variant === "rail") {
        const preferredRight = anchorRect.right + gap;
        const fallbackLeft = anchorRect.left - menuWidth - gap;
        left = preferredRight + menuWidth <= viewportWidth - margin ? preferredRight : fallbackLeft;
        top = anchorRect.top + 8;
      } else {
        left = anchorRect.left + anchorRect.width / 2 - menuWidth / 2;
        const belowTop = anchorRect.bottom + 6;
        const aboveTop = anchorRect.top - menuHeight - 6;
        top =
          belowTop + menuHeight <= viewportHeight - margin || aboveTop < margin ? belowTop : aboveTop;
      }

      left = Math.min(Math.max(left, margin), viewportWidth - menuWidth - margin);
      top = Math.min(Math.max(top, margin), viewportHeight - menuHeight - margin);

      setInsertMenuPosition((current) =>
        current && current.left === left && current.top === top ? current : { left, top }
      );
    };

    const scheduleUpdate = () => {
      window.cancelAnimationFrame(rafId);
      rafId = window.requestAnimationFrame(updatePosition);
    };

    scheduleUpdate();
    // Re-measure after first paint when the menu node has dimensions.
    window.requestAnimationFrame(scheduleUpdate);

    window.addEventListener("resize", scheduleUpdate);
    window.addEventListener("scroll", scheduleUpdate, true);

    return () => {
      window.cancelAnimationFrame(rafId);
      window.removeEventListener("resize", scheduleUpdate);
      window.removeEventListener("scroll", scheduleUpdate, true);
    };
  }, [openInsertMenu]);

  function toggleInsertMenu(nextMenu: OpenInsertMenu): void {
    setOpenInsertMenu((current) => (current?.key === nextMenu.key ? null : nextMenu));
  }

  function closeInsertMenu(): void {
    setOpenInsertMenu(null);
    setInsertMenuPosition(null);
  }

  function openInsertMenuFromEvent(
    event: MouseEvent<HTMLButtonElement>,
    config: Omit<OpenInsertMenu, "anchorEl">
  ): void {
    toggleInsertMenu({
      ...config,
      anchorEl: event.currentTarget,
    });
  }

  function handleToggleKeyDown(
    event: ReactKeyboardEvent<HTMLElement>,
    toggle: () => void
  ): void {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      toggle();
    }
  }

  function renderBlockInsertSlot(insertIndex: number): ReactNode {
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
            openInsertMenuFromEvent(event, {
              key: menuKey,
              title: "Add block",
              variant: "rail",
              items: [
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
              ],
            })
          }
        />
      </div>
    );
  }

  function renderComponentInsertSlot(blockId: string, insertIndex: number): ReactNode {
    const menuKey = `component:${blockId}:${insertIndex}`;
    const slotKey = `component-slot-${blockId}-${insertIndex}`;
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
          onClick={(event) =>
            openInsertMenuFromEvent(event, {
              key: menuKey,
              title: "Add component",
              variant: "inline",
              items: (["attention", "mlp", "norm", "activation"] as const).map((kind) => ({
                id: kind,
                label: labelForComponentKind(kind),
                onSelect: () => insertComponentAt(blockId, insertIndex, kind),
              })),
            })
          }
        />
      </div>
    );
  }

  function renderMlpStepInsertSlot(
    blockId: string,
    componentId: string,
    insertIndex: number
  ): ReactNode {
    const menuKey = `mlp-step:${blockId}:${componentId}:${insertIndex}`;
    const slotKey = `mlp-slot-${blockId}-${componentId}-${insertIndex}`;
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
        <span className="mlpStepInsertLabel" aria-hidden>
          + Add step
        </span>
        <button
          type="button"
          className="inlineInsertTrigger"
          aria-label={`Add MLP step at position ${insertIndex + 1}`}
          aria-haspopup="menu"
          aria-expanded={isOpen}
          onClick={(event) =>
            openInsertMenuFromEvent(event, {
              key: menuKey,
              title: "Add MLP step",
              variant: "inline",
              items: (["linear", "norm", "activation"] as const).map((kind) => ({
                id: kind,
                label: labelForMlpStepKind(kind),
                onSelect: () => insertMlpStepAt(blockId, componentId, insertIndex, kind),
              })),
            })
          }
        />
      </div>
    );
  }

  function renderNormFields(
    norm: NormConfig,
    onChange: (next: NormConfig) => void,
    idPrefix: string
  ): ReactNode {
    return (
      <div className="fieldGrid compact">
        <label className="fieldLabel" htmlFor={`${idPrefix}-norm-type`}>
          <span>Norm type</span>
          <select
            id={`${idPrefix}-norm-type`}
            value={norm.type}
            onChange={(event) => {
              if (event.target.value === "rmsnorm") {
                onChange({ type: "rmsnorm", learnable_gamma: true });
              } else {
                onChange({ type: "layernorm" });
              }
            }}
          >
            <option value="layernorm">layernorm</option>
            <option value="rmsnorm">rmsnorm</option>
          </select>
        </label>
        {norm.type === "rmsnorm" ? (
          <label className="toggleField" htmlFor={`${idPrefix}-learnable-gamma`}>
            <input
              id={`${idPrefix}-learnable-gamma`}
              type="checkbox"
              checked={norm.learnable_gamma}
              onChange={(event) =>
                onChange({ type: "rmsnorm", learnable_gamma: event.target.checked })
              }
            />
            <span>learnable</span>
          </label>
        ) : null}
      </div>
    );
  }

  function renderBlockCard(block: StudioBlock, blockIndex: number): ReactNode {
    return (
      <article className="blockCard">
        <div className="blockCardHead">
          <div className="blockCardTitleWrap">
            <div>
              <h3>Block {blockIndex + 1}</h3>
              <p>{block.components.length} component{block.components.length === 1 ? "" : "s"}</p>
            </div>
          </div>
          <div className="blockCardActions">
            <button
              type="button"
              className="iconButton"
              onClick={() => duplicateBlock(block.id)}
              title="Duplicate block"
              aria-label={`Duplicate block ${blockIndex + 1}`}
            >
              <FiCopy />
            </button>
            <button
              type="button"
              className="iconButton danger"
              onClick={() => deleteBlock(block.id)}
              title={documentState.blocks.length <= 1 ? "Keep at least one block" : "Delete block"}
              aria-label={`Delete block ${blockIndex + 1}`}
              disabled={documentState.blocks.length <= 1}
            >
              <FiTrash2 />
            </button>
          </div>
        </div>
        <div className="blockQuickMap" aria-label={`Block ${blockIndex + 1} sequence`}>
          {block.components.length === 0 ? (
            <span className="flowMiniChip isEmpty">Empty</span>
          ) : (
            block.components.map((component, componentIndex) => (
              <span
                key={`${block.id}-${component.id}-chip`}
                className={`flowMiniChip kind-${component.kind}`}
                title={`${componentIndex + 1}. ${labelForComponentKind(component.kind)} · ${summarizeComponent(component)}`}
              >
                {componentIndex + 1}. {labelForComponentKind(component.kind)}
              </span>
            ))
          )}
        </div>

        <div className="componentLane">
          {renderComponentInsertSlot(block.id, 0)}

          {block.components.length === 0 ? (
            <div className="emptyLaneHint">
              Click the insertion slot to choose the first component, or drag an existing component here.
            </div>
          ) : null}

          {block.components.map((component, componentIndex) => (
            <Fragment key={component.id}>
              <section
                className={`componentCard kind-${component.kind}${expandedComponentIds.has(component.id) ? "" : " isCollapsed"}`}
              >
                <div
                  className="componentCardHead isToggleable"
                  role="button"
                  tabIndex={0}
                  aria-expanded={expandedComponentIds.has(component.id)}
                  aria-label={`${expandedComponentIds.has(component.id) ? "Collapse" : "Expand"} ${labelForComponentKind(component.kind)} component settings`}
                  onClick={() => toggleExpandedComponent(component.id)}
                  onKeyDown={(event) => handleToggleKeyDown(event, () => toggleExpandedComponent(component.id))}
                >
                  <div
                    className="dragBadge"
                    draggable
                    onDragStart={(event) => beginDragComponent(event, block.id, component.id)}
                    onDragEnd={clearDragState}
                    onClick={(event) => event.stopPropagation()}
                    title={`Drag ${labelForComponentKind(component.kind)} component`}
                    aria-hidden
                  >
                    <FiMove />
                  </div>
                  <div className="componentMeta">
                    <span className="componentTag">{labelForComponentKind(component.kind)}</span>
                    <span className="componentSummary">{summarizeComponent(component)}</span>
                  </div>
                  <div className="componentHeadActions">
                    <span className="componentToggleGlyph" aria-hidden>
                      {expandedComponentIds.has(component.id) ? <FiChevronDown /> : <FiChevronRight />}
                    </span>
                    <button
                      type="button"
                      className="iconButton danger"
                      onClick={(event) => {
                        event.stopPropagation();
                        removeComponent(block.id, component.id);
                      }}
                      aria-label="Remove component"
                      title="Remove component"
                    >
                      <FiTrash2 />
                    </button>
                  </div>
                </div>

                {!expandedComponentIds.has(component.id) && component.kind === "mlp" ? (
                  <div className="componentCollapsedTrail" aria-label="MLP sequence summary">
                    {component.mlp.sequence.map((step, stepIndex) => (
                      <span
                        key={`${component.id}-${step.id}-trail`}
                        className={`miniStepBadge kind-${step.kind}`}
                        title={`${stepIndex + 1}. ${labelForMlpStepKind(step.kind)} · ${summarizeMlpStep(step)}`}
                      >
                        {stepIndex + 1}. {labelForMlpStepKind(step.kind)}
                      </span>
                    ))}
                  </div>
                ) : null}

                {expandedComponentIds.has(component.id) ? (
                  <div className="componentBody">
                    {component.kind === "attention" ? (
                      <div className="fieldGrid compact">
                        <label
                          className="fieldLabel"
                          htmlFor={`${componentDomIdPrefix(blockIndex, componentIndex)}-n-head`}
                        >
                          <span>n_head</span>
                          <input
                            id={`${componentDomIdPrefix(blockIndex, componentIndex)}-n-head`}
                            type="number"
                            min={1}
                            step={1}
                            value={integerInputValue(component.attention.n_head)}
                            onChange={(event) =>
                              updateComponent(block.id, component.id, (current) =>
                                current.kind !== "attention"
                                  ? current
                                  : {
                                      ...current,
                                      attention: {
                                        ...current.attention,
                                        n_head: parseIntegerInput(
                                          event.target.value,
                                          current.attention.n_head
                                        ),
                                      },
                                    }
                              )
                            }
                          />
                        </label>
                        <label
                          className="fieldLabel"
                          htmlFor={`${componentDomIdPrefix(blockIndex, componentIndex)}-n-kv-head`}
                        >
                          <span>n_kv_head</span>
                          <input
                            id={`${componentDomIdPrefix(blockIndex, componentIndex)}-n-kv-head`}
                            type="number"
                            min={1}
                            step={1}
                            value={integerInputValue(component.attention.n_kv_head)}
                            onChange={(event) =>
                              updateComponent(block.id, component.id, (current) =>
                                current.kind !== "attention"
                                  ? current
                                  : {
                                      ...current,
                                      attention: {
                                        ...current.attention,
                                        n_kv_head: parseIntegerInput(
                                          event.target.value,
                                          current.attention.n_kv_head
                                        ),
                                      },
                                    }
                              )
                            }
                          />
                        </label>
                      </div>
                    ) : null}

                    {component.kind === "norm"
                      ? renderNormFields(component.norm, (nextNorm) => {
                          updateComponent(block.id, component.id, (current) =>
                            current.kind !== "norm" ? current : { ...current, norm: nextNorm }
                          );
                        }, componentDomIdPrefix(blockIndex, componentIndex))
                      : null}

                    {component.kind === "activation" ? (
                      <div className="fieldGrid compact">
                        <label
                          className="fieldLabel"
                          htmlFor={`${componentDomIdPrefix(blockIndex, componentIndex)}-activation-type`}
                        >
                          <span>Activation</span>
                          <select
                            id={`${componentDomIdPrefix(blockIndex, componentIndex)}-activation-type`}
                            value={component.activation.type}
                            onChange={(event) =>
                              updateComponent(block.id, component.id, (current) =>
                                current.kind !== "activation"
                                  ? current
                                  : {
                                      ...current,
                                      activation: {
                                        type: event.target.value as ActivationType,
                                      },
                                    }
                              )
                            }
                          >
                            {ACTIVATION_TYPES.map((type) => (
                              <option key={type} value={type}>
                                {type}
                              </option>
                            ))}
                          </select>
                        </label>
                      </div>
                    ) : null}

                    {component.kind === "mlp" ? (
                      <div className="mlpEditor">
                        <div className="fieldGrid compact mlpEditorFields">
                          <label
                            className="fieldLabel mlpMultiplierField"
                            htmlFor={`${componentDomIdPrefix(blockIndex, componentIndex)}-multiplier`}
                          >
                            <span>multiplier</span>
                            <input
                              id={`${componentDomIdPrefix(blockIndex, componentIndex)}-multiplier`}
                              type="number"
                              min={0.001}
                              step="any"
                              value={numberInputValue(component.mlp.multiplier)}
                              onChange={(event) =>
                                updateComponent(block.id, component.id, (current) =>
                                  current.kind !== "mlp"
                                    ? current
                                    : {
                                        ...current,
                                        mlp: {
                                          ...current.mlp,
                                          multiplier: parseNumberInput(
                                            event.target.value,
                                            current.mlp.multiplier
                                          ),
                                        },
                                      }
                                )
                              }
                            />
                          </label>
                        </div>

                        <div className="mlpSequenceShell">
                          <div className="mlpSequenceHead">
                            <div className="miniLabel">MLP steps</div>
                            <span className="mlpSequenceCount">
                              {component.mlp.sequence.length} step
                              {component.mlp.sequence.length === 1 ? "" : "s"}
                            </span>
                          </div>
                          <div className="mlpSequenceList" role="list" aria-label="MLP step sequence">
                            {renderMlpStepInsertSlot(block.id, component.id, 0)}
                            {component.mlp.sequence.length === 0 ? (
                              <div className="mlpSequenceEmpty" role="listitem">
                                <div className="emptyLaneHint compact">
                                  Add a step, or drag an existing step here.
                                </div>
                              </div>
                            ) : null}
                            {component.mlp.sequence.map((step, stepIndex) => (
                              <Fragment key={step.id}>
                                <div
                                  className={`mlpStepCard kind-${step.kind}${expandedMlpStepIds.has(step.id) ? "" : " isCollapsed"}`}
                                  role="listitem"
                                >
                                  <div
                                    className="componentCardHead isToggleable"
                                    role="button"
                                    tabIndex={0}
                                    aria-expanded={expandedMlpStepIds.has(step.id)}
                                    aria-label={`${expandedMlpStepIds.has(step.id) ? "Collapse" : "Expand"} MLP step settings`}
                                    onClick={() => toggleExpandedMlpStep(step.id)}
                                    onKeyDown={(event) =>
                                      handleToggleKeyDown(event, () => toggleExpandedMlpStep(step.id))
                                    }
                                  >
                                    <div
                                      className="dragBadge"
                                      draggable
                                      onDragStart={(event) =>
                                        beginDragMlpStep(event, block.id, component.id, step.id)
                                      }
                                      onDragEnd={clearDragState}
                                      onClick={(event) => event.stopPropagation()}
                                      title={`Drag ${labelForMlpStepKind(step.kind)} step`}
                                      aria-hidden
                                    >
                                      <FiMove />
                                    </div>
                                    <div className="componentMeta">
                                      <span className="componentTag">
                                        {stepIndex + 1}. {labelForMlpStepKind(step.kind)}
                                      </span>
                                      <span className="componentSummary">{summarizeMlpStep(step)}</span>
                                    </div>
                                    <div className="componentHeadActions">
                                      <span className="componentToggleGlyph" aria-hidden>
                                        {expandedMlpStepIds.has(step.id) ? (
                                          <FiChevronDown />
                                        ) : (
                                          <FiChevronRight />
                                        )}
                                      </span>
                                      <button
                                        type="button"
                                        className="iconButton danger"
                                        onClick={(event) => {
                                          event.stopPropagation();
                                          removeMlpStep(block.id, component.id, step.id);
                                        }}
                                        aria-label="Remove MLP step"
                                        title="Remove MLP step"
                                      >
                                        <FiTrash2 />
                                      </button>
                                    </div>
                                  </div>
                                  {expandedMlpStepIds.has(step.id) ? (
                                    <div className="componentBody">
                                      {step.kind === "linear" ? (
                                        <label
                                          className="toggleField"
                                          htmlFor={`${mlpStepDomIdPrefix(blockIndex, componentIndex, stepIndex)}-linear-bias`}
                                        >
                                          <input
                                            id={`${mlpStepDomIdPrefix(blockIndex, componentIndex, stepIndex)}-linear-bias`}
                                            type="checkbox"
                                            checked={step.linear.bias}
                                            onChange={(event) =>
                                              updateMlpStep(
                                                block.id,
                                                component.id,
                                                step.id,
                                                (current) =>
                                                  current.kind !== "linear"
                                                    ? current
                                                    : {
                                                        ...current,
                                                        linear: { bias: event.target.checked },
                                                      }
                                              )
                                            }
                                          />
                                          <span>bias</span>
                                        </label>
                                      ) : null}

                                      {step.kind === "norm"
                                        ? renderNormFields(step.norm, (nextNorm) => {
                                            updateMlpStep(
                                              block.id,
                                              component.id,
                                              step.id,
                                              (current) =>
                                                current.kind !== "norm"
                                                  ? current
                                                  : { ...current, norm: nextNorm }
                                            );
                                          }, mlpStepDomIdPrefix(blockIndex, componentIndex, stepIndex))
                                        : null}

                                      {step.kind === "activation" ? (
                                        <div className="fieldGrid compact">
                                          <label
                                            className="fieldLabel"
                                            htmlFor={`${mlpStepDomIdPrefix(blockIndex, componentIndex, stepIndex)}-activation-type`}
                                          >
                                            <span>Activation</span>
                                            <select
                                              id={`${mlpStepDomIdPrefix(blockIndex, componentIndex, stepIndex)}-activation-type`}
                                              value={step.activation.type}
                                              onChange={(event) =>
                                                updateMlpStep(
                                                  block.id,
                                                  component.id,
                                                  step.id,
                                                  (current) =>
                                                    current.kind !== "activation"
                                                      ? current
                                                      : {
                                                          ...current,
                                                          activation: {
                                                            type: event.target.value as ActivationType,
                                                          },
                                                        }
                                                )
                                              }
                                            >
                                              {ACTIVATION_TYPES.map((type) => (
                                                <option key={type} value={type}>
                                                  {type}
                                                </option>
                                              ))}
                                            </select>
                                          </label>
                                        </div>
                                      ) : null}
                                    </div>
                                  ) : null}
                                </div>
                                {renderMlpStepInsertSlot(block.id, component.id, stepIndex + 1)}
                              </Fragment>
                            ))}
                          </div>
                        </div>
                      </div>
                    ) : null}
                  </div>
                ) : null}
              </section>
              {renderComponentInsertSlot(block.id, componentIndex + 1)}
            </Fragment>
          ))}
        </div>
      </article>
    );
  }

  return (
    <>
      <section id="block-builder" className="panelCard builderPanel">
        <div className="panelHead">
          <div>
            <p className="panelEyebrow">Visual Builder</p>
            <h2>Horizontal block canvas</h2>
            <p className="panelCopy">
              Scroll horizontally for model depth and vertically for details. Click insertion slots to add blocks/components/MLP steps, and drag existing components or steps to reorder them.
            </p>
          </div>
          <div className="actionCluster">
            <button type="button" className="buttonGhost" onClick={collapseAllCanvasNodes}>
              <FiChevronRight /> Collapse all
            </button>
            <button type="button" className="buttonGhost" onClick={expandAllCanvasNodes}>
              <FiChevronDown /> Expand all
            </button>
            <button type="button" className="buttonGhost" onClick={addBlock}>
              <FiPlus /> Add block
            </button>
          </div>
        </div>

        <div className="builderCanvasToolbar">
          <div className="builderCanvasHint">
            <span className="builderCanvasHintDot" aria-hidden />
            Canvas scrolls in both directions. Blocks are columns; components stay attached inside each block.
          </div>
          <div className="builderCanvasStats" aria-label="Builder canvas statistics">
            <span>{metrics.blockCount} blocks</span>
            <span>{metrics.componentCount} components</span>
            <span>{metrics.mlpStepCount} MLP steps</span>
          </div>
        </div>

        <div className="blockCanvasViewport" role="region" aria-label="Horizontal model block canvas">
          <div className="blockCanvas">
          {renderBlockInsertSlot(0)}

          {consecutiveBlockGroups.map((group) => {
            const groupBlocks = documentState.blocks.slice(group.startIndex, group.endIndex + 1);
            const representativeBlock = groupBlocks[0];
            const isRepeatedGroup = group.count > 1;
            const groupExpanded = isRepeatedGroup && expandedBlockGroupKeys.has(group.key);

            return (
              <Fragment key={isRepeatedGroup ? group.key : representativeBlock.id}>
                {isRepeatedGroup ? (
                  <section
                    className={`blockGroupCard${groupExpanded ? " isExpanded" : ""}`}
                    aria-label={`Identical block group spanning blocks ${group.startIndex + 1} through ${group.endIndex + 1}`}
                  >
                    <div
                      className="blockGroupHead isToggleable"
                      role="button"
                      tabIndex={0}
                      aria-expanded={groupExpanded}
                      aria-label={groupExpanded ? "Collapse identical block group" : "Expand identical block group"}
                      onClick={() => toggleExpandedBlockGroup(group.key)}
                      onKeyDown={(event) =>
                        handleToggleKeyDown(event, () => toggleExpandedBlockGroup(group.key))
                      }
                    >
                      <div className="blockGroupTitleWrap">
                        <div className="blockGroupBadge" aria-hidden>
                          <FiLayers />
                        </div>
                        <div>
                          <h3>
                            Blocks {group.startIndex + 1}-{group.endIndex + 1}
                          </h3>
                          <p>{group.count} identical blocks</p>
                        </div>
                      </div>
                    </div>

                    <div className="blockQuickMap" aria-label="Repeated block structure preview">
                      {representativeBlock.components.length === 0 ? (
                        <span className="flowMiniChip isEmpty">Empty</span>
                      ) : (
                        representativeBlock.components.map((component, componentIndex) => (
                          <span
                            key={`${representativeBlock.id}-${component.id}-group-chip`}
                            className={`flowMiniChip kind-${component.kind}`}
                            title={`${componentIndex + 1}. ${labelForComponentKind(component.kind)} · ${summarizeComponent(component)}`}
                          >
                            {componentIndex + 1}. {labelForComponentKind(component.kind)}
                          </span>
                        ))
                      )}
                    </div>

                    {!groupExpanded ? (
                      <p className="blockGroupHint">
                        Expand to edit or reorder individual blocks inside this repeated run.
                      </p>
                    ) : (
                      <div className="blockGroupTrack">
                        {groupBlocks.map((block, offset) => {
                          const absoluteIndex = group.startIndex + offset;
                          const isLastInGroup = offset === groupBlocks.length - 1;
                          return (
                            <Fragment key={block.id}>
                              {renderBlockCard(block, absoluteIndex)}
                              {!isLastInGroup ? (
                                renderBlockInsertSlot(absoluteIndex + 1)
                              ) : null}
                            </Fragment>
                          );
                        })}
                      </div>
                    )}
                  </section>
                ) : (
                  renderBlockCard(representativeBlock, group.startIndex)
                )}
                {renderBlockInsertSlot(group.endIndex + 1)}
              </Fragment>
            );
          })}
          </div>
        </div>
      </section>
      {openInsertMenu && typeof document !== "undefined"
        ? createPortal(
            <div
              ref={insertMenuRef}
              className={`blockInsertMenu insertMenuPortal variant-${openInsertMenu.variant}`}
              data-insert-slot
              role="menu"
              aria-label={openInsertMenu.title}
              style={
                insertMenuPosition
                  ? { left: insertMenuPosition.left, top: insertMenuPosition.top }
                  : { left: -9999, top: -9999 }
              }
            >
              <div className="blockInsertMenuHeader">{openInsertMenu.title}</div>
              {openInsertMenu.items.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  className="blockInsertMenuButton"
                  role="menuitem"
                  onClick={() => {
                    item.onSelect();
                    closeInsertMenu();
                  }}
                >
                  <span className="blockInsertMenuButtonTitle">{item.label}</span>
                </button>
              ))}
            </div>,
            document.body
          )
        : null}
    </>
  );
}
