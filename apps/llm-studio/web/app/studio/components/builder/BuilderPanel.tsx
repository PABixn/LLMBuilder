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
  labelForActivationType,
  labelForComponentKind,
  labelForMlpStepKind,
  labelForNormType,
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
  const quickMapChipLimit = 4;
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
            <option value="layernorm">{labelForNormType("layernorm")}</option>
            <option value="rmsnorm">{labelForNormType("rmsnorm")}</option>
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
            <span>Learnable</span>
          </label>
        ) : null}
      </div>
    );
  }

  function renderInlineNormControls(
    norm: NormConfig,
    onChange: (next: NormConfig) => void,
    idPrefix: string
  ): ReactNode {
    return (
      <div className="componentHeadInlineFields" onClick={(event) => event.stopPropagation()}>
        <label className="headerInlineField" htmlFor={`${idPrefix}-norm-type`}>
          <select
            id={`${idPrefix}-norm-type`}
            aria-label="Norm type"
            title="Norm type"
            value={norm.type}
            onChange={(event) => {
              if (event.target.value === "rmsnorm") {
                onChange({ type: "rmsnorm", learnable_gamma: true });
              } else {
                onChange({ type: "layernorm" });
              }
            }}
          >
            <option value="layernorm">{labelForNormType("layernorm")}</option>
            <option value="rmsnorm">{labelForNormType("rmsnorm")}</option>
          </select>
        </label>
        {norm.type === "rmsnorm" ? (
          <label className="headerInlineToggle" htmlFor={`${idPrefix}-learnable-gamma`}>
            <input
              id={`${idPrefix}-learnable-gamma`}
              type="checkbox"
              aria-label="RMSNorm learnable gamma"
              title="RMSNorm learnable gamma"
              checked={norm.learnable_gamma}
              onChange={(event) =>
                onChange({ type: "rmsnorm", learnable_gamma: event.target.checked })
              }
            />
            <span>Learnable</span>
          </label>
        ) : null}
      </div>
    );
  }

  function renderInlineActivationControls(
    activationType: ActivationType,
    onChange: (next: ActivationType) => void,
    idPrefix: string
  ): ReactNode {
    return (
      <div className="componentHeadInlineFields" onClick={(event) => event.stopPropagation()}>
        <label className="headerInlineField" htmlFor={`${idPrefix}-activation-type`}>
          <select
            id={`${idPrefix}-activation-type`}
            aria-label="Activation type"
            title="Activation type"
            value={activationType}
            onChange={(event) => onChange(event.target.value as ActivationType)}
          >
            {ACTIVATION_TYPES.map((type) => (
              <option key={type} value={type}>
                {labelForActivationType(type)}
              </option>
            ))}
          </select>
        </label>
      </div>
    );
  }

  function renderInlineLinearControls(
    bias: boolean,
    onChange: (nextBias: boolean) => void,
    idPrefix: string
  ): ReactNode {
    return (
      <div className="componentHeadInlineFields" onClick={(event) => event.stopPropagation()}>
        <label className="headerInlineToggle" htmlFor={`${idPrefix}-linear-bias`}>
          <input
            id={`${idPrefix}-linear-bias`}
            type="checkbox"
            aria-label="Linear bias"
            title="Linear bias"
            checked={bias}
            onChange={(event) => onChange(event.target.checked)}
          />
          <span>Bias</span>
        </label>
      </div>
    );
  }

  function renderQuickMapChips(components: StudioComponent[], chipKeyPrefix: string): ReactNode {
    if (components.length === 0) {
      return <span className="flowMiniChip isEmpty">Empty</span>;
    }

    const visibleComponents = components.slice(0, quickMapChipLimit);
    const hiddenCount = Math.max(0, components.length - visibleComponents.length);

    return (
      <>
        {visibleComponents.map((component, componentIndex) => (
          <span
            key={`${chipKeyPrefix}-${component.id}-chip`}
            className={`flowMiniChip kind-${component.kind}`}
            title={`${componentIndex + 1}. ${labelForComponentKind(component.kind)} · ${summarizeComponent(component)}`}
          >
            {componentIndex + 1}. {labelForComponentKind(component.kind)}
          </span>
        ))}
        {hiddenCount > 0 ? (
          <span className="flowMiniChip isMeta" title={`${components.length} components total`}>
            +{hiddenCount} more
          </span>
        ) : null}
      </>
    );
  }

  function isInlineSimpleComponent(component: StudioComponent): boolean {
    return component.kind === "norm" || component.kind === "activation";
  }

  function isInlineSimpleMlpStep(_step: StudioMlpStep): boolean {
    return true;
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
              aria-label={`Add component to block ${blockIndex + 1}`}
              aria-haspopup="menu"
              aria-expanded={openInsertMenu?.key === `component:${block.id}:${block.components.length}`}
              title="Add component"
              onClick={(event) =>
                openInsertMenuFromEvent(event, {
                  key: `component:${block.id}:${block.components.length}`,
                  title: "Add component",
                  variant: "inline",
                  items: (["attention", "mlp", "norm", "activation"] as const).map((kind) => ({
                    id: kind,
                    label: labelForComponentKind(kind),
                    onSelect: () => insertComponentAt(block.id, block.components.length, kind),
                  })),
                })
              }
            >
              <FiPlus />
            </button>
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
          {renderQuickMapChips(block.components, block.id)}
        </div>

        <div className="componentLane">
            {block.components.length === 0 ? (
              <div className="emptyLaneHint">
                Click an insert slot or drag a component here.
              </div>
            ) : null}

            {block.components.map((component, componentIndex) => {
              const componentIsInlineSimple = isInlineSimpleComponent(component);
              const componentIsExpanded =
                componentIsInlineSimple || expandedComponentIds.has(component.id);

              return (
                <Fragment key={component.id}>
                  <section
                    className={`componentCard kind-${component.kind}${componentIsInlineSimple ? " isInlineSimple" : ""}${!componentIsExpanded ? " isCollapsed" : ""}`}
                  >
                <div
                  className={`componentCardHead${componentIsInlineSimple ? "" : " isToggleable"}`}
                  role={componentIsInlineSimple ? undefined : "button"}
                  tabIndex={componentIsInlineSimple ? undefined : 0}
                  aria-expanded={componentIsInlineSimple ? undefined : componentIsExpanded}
                  aria-label={
                    componentIsInlineSimple
                      ? undefined
                      : `${componentIsExpanded ? "Collapse" : "Expand"} ${labelForComponentKind(component.kind)} component settings`
                  }
                  onClick={componentIsInlineSimple ? undefined : () => toggleExpandedComponent(component.id)}
                  onKeyDown={
                    componentIsInlineSimple
                      ? undefined
                      : (event) =>
                          handleToggleKeyDown(event, () => toggleExpandedComponent(component.id))
                  }
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
                  {component.kind === "norm"
                    ? renderInlineNormControls(
                        component.norm,
                        (nextNorm) => {
                          updateComponent(block.id, component.id, (current) =>
                            current.kind !== "norm" ? current : { ...current, norm: nextNorm }
                          );
                        },
                        componentDomIdPrefix(blockIndex, componentIndex)
                      )
                    : null}
                  {component.kind === "activation"
                    ? renderInlineActivationControls(
                        component.activation.type,
                        (nextType) =>
                          updateComponent(block.id, component.id, (current) =>
                            current.kind !== "activation"
                              ? current
                              : { ...current, activation: { type: nextType } }
                          ),
                        componentDomIdPrefix(blockIndex, componentIndex)
                      )
                    : null}
                  <div className="componentHeadActions">
                    {!componentIsInlineSimple ? (
                      <span className="componentToggleGlyph" aria-hidden>
                        {componentIsExpanded ? <FiChevronDown /> : <FiChevronRight />}
                      </span>
                    ) : null}
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

                {!componentIsExpanded && component.kind === "mlp" ? (
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

                {componentIsExpanded && !componentIsInlineSimple ? (
                  <div className="componentBody">
                    {component.kind === "attention" ? (
                      <div className="fieldGrid compact">
                        <label
                          className="fieldLabel"
                          htmlFor={`${componentDomIdPrefix(blockIndex, componentIndex)}-n-head`}
                        >
                          <span>Heads</span>
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
                          <span>KV Heads</span>
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
                            <span>Multiplier</span>
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
                            <div className="mlpSequenceTitleWrap">
                              <div className="miniLabel">MLP steps</div>
                              <span className="mlpSequenceCount">
                                {component.mlp.sequence.length} step
                                {component.mlp.sequence.length === 1 ? "" : "s"}
                              </span>
                            </div>
                            <button
                              type="button"
                              className="mlpSequenceAddButton"
                              aria-label="Add MLP step at end"
                              aria-haspopup="menu"
                              aria-expanded={
                                openInsertMenu?.key ===
                                `mlp-step:${block.id}:${component.id}:${component.mlp.sequence.length}`
                              }
                              onClick={(event) =>
                                openInsertMenuFromEvent(event, {
                                  key: `mlp-step:${block.id}:${component.id}:${component.mlp.sequence.length}`,
                                  title: "Add MLP step",
                                  variant: "inline",
                                  items: (["linear", "norm", "activation"] as const).map((kind) => ({
                                    id: kind,
                                    label: labelForMlpStepKind(kind),
                                    onSelect: () =>
                                      insertMlpStepAt(
                                        block.id,
                                        component.id,
                                        component.mlp.sequence.length,
                                        kind
                                      ),
                                  })),
                                })
                              }
                            >
                              <FiPlus />
                            </button>
                          </div>
                          <div className="mlpSequenceList" role="list" aria-label="MLP step sequence">
                            {component.mlp.sequence.length === 0 ? (
                              <div className="mlpSequenceEmpty" role="listitem">
                                <div className="emptyLaneHint compact">
                                  Add a step or drag one here.
                                </div>
                              </div>
                            ) : null}
                            {component.mlp.sequence.map((step, stepIndex) => {
                              const stepIsInlineSimple = isInlineSimpleMlpStep(step);
                              const stepIsExpanded =
                                stepIsInlineSimple || expandedMlpStepIds.has(step.id);

                              return (
                                <Fragment key={step.id}>
                                <div
                                  className={`mlpStepCard kind-${step.kind}${stepIsInlineSimple ? " isInlineSimple" : ""}${!stepIsExpanded ? " isCollapsed" : ""}`}
                                  role="listitem"
                                >
                                  <div
                                    className={`componentCardHead${stepIsInlineSimple ? "" : " isToggleable"}`}
                                    role={stepIsInlineSimple ? undefined : "button"}
                                    tabIndex={stepIsInlineSimple ? undefined : 0}
                                    aria-expanded={stepIsInlineSimple ? undefined : stepIsExpanded}
                                    aria-label={
                                      stepIsInlineSimple
                                        ? undefined
                                        : `${stepIsExpanded ? "Collapse" : "Expand"} MLP step settings`
                                    }
                                    onClick={stepIsInlineSimple ? undefined : () => toggleExpandedMlpStep(step.id)}
                                    onKeyDown={
                                      stepIsInlineSimple
                                        ? undefined
                                        : (event) =>
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
                                    {step.kind === "linear"
                                      ? renderInlineLinearControls(
                                          step.linear.bias,
                                          (nextBias) =>
                                            updateMlpStep(
                                              block.id,
                                              component.id,
                                              step.id,
                                              (current) =>
                                                current.kind !== "linear"
                                                  ? current
                                                  : { ...current, linear: { bias: nextBias } }
                                            ),
                                          `${mlpStepDomIdPrefix(blockIndex, componentIndex, stepIndex)}`
                                        )
                                      : null}
                                    {step.kind === "norm"
                                      ? renderInlineNormControls(
                                          step.norm,
                                          (nextNorm) =>
                                            updateMlpStep(
                                              block.id,
                                              component.id,
                                              step.id,
                                              (current) =>
                                                current.kind !== "norm"
                                                  ? current
                                                  : { ...current, norm: nextNorm }
                                            ),
                                          mlpStepDomIdPrefix(blockIndex, componentIndex, stepIndex)
                                        )
                                      : null}
                                    {step.kind === "activation"
                                      ? renderInlineActivationControls(
                                          step.activation.type,
                                          (nextType) =>
                                            updateMlpStep(
                                              block.id,
                                              component.id,
                                              step.id,
                                              (current) =>
                                                current.kind !== "activation"
                                                  ? current
                                                  : {
                                                      ...current,
                                                      activation: { type: nextType },
                                                    }
                                            ),
                                          mlpStepDomIdPrefix(blockIndex, componentIndex, stepIndex)
                                        )
                                      : null}
                                    <div className="componentHeadActions">
                                      {!stepIsInlineSimple ? (
                                        <span className="componentToggleGlyph" aria-hidden>
                                          {stepIsExpanded ? <FiChevronDown /> : <FiChevronRight />}
                                        </span>
                                      ) : null}
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
                                  {stepIsExpanded && !stepIsInlineSimple ? (
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
                                          <span>Bias</span>
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
                                {labelForActivationType(type)}
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
                              );
                            })}
                          </div>
                        </div>
                      </div>
                    ) : null}
                  </div>
                ) : null}
                  </section>
                  {renderComponentInsertSlot(block.id, componentIndex + 1)}
                </Fragment>
              );
            })}
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
            <h2>Visual designer</h2>
            <p className="panelCopy">
              Build depth horizontally. Use insert slots to add blocks/components/MLP steps, then drag to reorder.
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
            Two-axis canvas: blocks are columns, components stay nested inside each block.
          </div>
          <div className="builderCanvasStats" aria-label="Builder canvas statistics">
            <span>{metrics.blockCount} blocks</span>
            <span>{metrics.componentCount} components</span>
            <span>{metrics.mlpStepCount} MLP steps</span>
          </div>
        </div>

        <div className="blockCanvasViewport" role="region" aria-label="Horizontal model block canvas">
          <div className="blockCanvas">
            {consecutiveBlockGroups.map((group) => {
              const groupBlocks = documentState.blocks.slice(group.startIndex, group.endIndex + 1);
              const representativeBlock = groupBlocks[0];
              const isRepeatedGroup = group.count > 1;
              const groupExpanded = isRepeatedGroup && expandedBlockGroupKeys.has(group.key);

              return (
                <Fragment key={isRepeatedGroup ? group.key : representativeBlock.id}>
                  {isRepeatedGroup ? (
                    groupExpanded ? (
                      <section
                        className="blockGroupCard isExpanded"
                        aria-label={`Identical block group spanning blocks ${group.startIndex + 1} through ${group.endIndex + 1}`}
                      >
                        <div
                          className="blockGroupHead isToggleable"
                          role="button"
                          tabIndex={0}
                          aria-expanded={groupExpanded}
                          aria-label="Collapse identical block group"
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
                            </div>
                          </div>
                        </div>

                        <div className="blockQuickMap" aria-label="Repeated block structure preview">
                          {renderQuickMapChips(
                            representativeBlock.components,
                            `${representativeBlock.id}-group`
                          )}
                        </div>

                        <div className="blockGroupTrack">
                          {groupBlocks.map((block, offset) => {
                            const absoluteIndex = group.startIndex + offset;
                            const isLastInGroup = offset === groupBlocks.length - 1;
                            return (
                              <Fragment key={block.id}>
                                {renderBlockCard(block, absoluteIndex)}
                                {!isLastInGroup ? renderBlockInsertSlot(absoluteIndex + 1) : null}
                              </Fragment>
                            );
                          })}
                        </div>
                      </section>
                    ) : (
                      <section
                        className="blockGroupCard blockGroupCollapsedPreview"
                        aria-label={`Collapsed identical block group spanning blocks ${group.startIndex + 1} through ${group.endIndex + 1}`}
                      >
                        <button
                          type="button"
                          className="blockGroupCollapsedExpand"
                          aria-expanded={false}
                          aria-label={`Expand identical block group (${group.count} blocks)`}
                          title={`Expand ${group.count} identical blocks`}
                          onClick={() => toggleExpandedBlockGroup(group.key)}
                        >
                          <span className="blockGroupCountBadge">×{group.count}</span>
                          <span className="blockGroupCollapsedExpandLabel">Expand Group</span>
                          <FiChevronRight aria-hidden />
                        </button>
                        {renderBlockCard(representativeBlock, group.startIndex)}
                      </section>
                    )
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
