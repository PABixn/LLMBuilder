import type { KeyboardEvent as ReactKeyboardEvent, MouseEvent, ReactNode } from "react";
import { Fragment } from "react";
import {
  FiChevronDown,
  FiChevronRight,
  FiCopy,
  FiLayers,
  FiMove,
  FiPlus,
  FiTrash2,
} from "react-icons/fi";

import { ACTIVATION_TYPES, type ActivationType } from "../../../../lib/defaults";

import type { StudioBlock, StudioComponent, StudioMlpStep } from "../../types";
import {
  labelForActivationType,
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
import {
  isInlineSimpleComponent,
  isInlineSimpleMlpStep,
  renderInlineActivationControls,
  renderInlineLinearControls,
  renderInlineNormControls,
  renderNormFields,
  renderQuickMapChips,
} from "./BuilderControls";
import {
  BlockInsertSlot,
  ComponentInsertSlot,
  MlpStepInsertSlot,
} from "./BuilderInsertSlots";
import type { BuilderPanelProps, OpenInsertMenu } from "./types";

type BuilderCanvasContentProps = BuilderPanelProps & {
  openInsertMenu: OpenInsertMenu | null;
  openInsertMenuFromEvent: (
    event: MouseEvent<HTMLButtonElement>,
    config: Omit<OpenInsertMenu, "anchorEl">
  ) => void;
  handleToggleKeyDown: (
    event: ReactKeyboardEvent<HTMLElement>,
    toggle: () => void
  ) => void;
};

export function BuilderCanvasContent({
  documentState,
  consecutiveBlockGroups,
  dragOverKey,
  expandedComponentIds,
  expandedMlpStepIds,
  expandedBlockGroupKeys,
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
  handleDropAtHighlightedSlot,
  openInsertMenu,
  openInsertMenuFromEvent,
  handleToggleKeyDown,
}: BuilderCanvasContentProps) {
  function renderBlockInsertSlot(insertIndex: number): ReactNode {
    return (
      <BlockInsertSlot
        insertIndex={insertIndex}
        insertBlockAt={insertBlockAt}
        openInsertMenu={openInsertMenu}
        openInsertMenuFromEvent={openInsertMenuFromEvent}
        dragOverKey={dragOverKey}
      />
    );
  }

  function renderComponentInsertSlot(blockId: string, insertIndex: number): ReactNode {
    return (
      <ComponentInsertSlot
        blockId={blockId}
        insertIndex={insertIndex}
        openInsertMenu={openInsertMenu}
        openInsertMenuFromEvent={openInsertMenuFromEvent}
        dragOverKey={dragOverKey}
        markDropTarget={markDropTarget}
        handleDropComponent={handleDropComponent}
        insertComponentAt={insertComponentAt}
      />
    );
  }

  function renderMlpStepInsertSlot(
    blockId: string,
    componentId: string,
    insertIndex: number
  ): ReactNode {
    return (
      <MlpStepInsertSlot
        blockId={blockId}
        componentId={componentId}
        insertIndex={insertIndex}
        openInsertMenu={openInsertMenu}
        openInsertMenuFromEvent={openInsertMenuFromEvent}
        dragOverKey={dragOverKey}
        markDropTarget={markDropTarget}
        handleDropMlpStep={handleDropMlpStep}
        insertMlpStepAt={insertMlpStepAt}
      />
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

        <div
          className="componentLane"
          onDragOverCapture={(event) => {
            if (!dragOverKey) {
              return;
            }
            event.preventDefault();
            event.dataTransfer.dropEffect = "move";
          }}
          onDropCapture={handleDropAtHighlightedSlot}
        >
          {block.components.length === 0 ? (
            <div className="emptyLaneHint">Click an insert slot or drag a component here.</div>
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
                    onClick={
                      componentIsInlineSimple
                        ? undefined
                        : () => toggleExpandedComponent(component.id)
                    }
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
                        ? renderNormFields(
                            component.norm,
                            (nextNorm) => {
                              updateComponent(block.id, component.id, (current) =>
                                current.kind !== "norm" ? current : { ...current, norm: nextNorm }
                              );
                            },
                            componentDomIdPrefix(blockIndex, componentIndex)
                          )
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
                                        onClick={
                                          stepIsInlineSimple
                                            ? undefined
                                            : () => toggleExpandedMlpStep(step.id)
                                        }
                                        onKeyDown={
                                          stepIsInlineSimple
                                            ? undefined
                                            : (event) =>
                                                handleToggleKeyDown(event, () =>
                                                  toggleExpandedMlpStep(step.id)
                                                )
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
                                          <span className="componentSummary">
                                            {summarizeMlpStep(step)}
                                          </span>
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
                                                            linear: {
                                                              bias: event.target.checked,
                                                            },
                                                          }
                                                  )
                                                }
                                              />
                                              <span>Bias</span>
                                            </label>
                                          ) : null}

                                          {step.kind === "norm"
                                            ? renderNormFields(
                                                step.norm,
                                                (nextNorm) => {
                                                  updateMlpStep(
                                                    block.id,
                                                    component.id,
                                                    step.id,
                                                    (current) =>
                                                      current.kind !== "norm"
                                                        ? current
                                                        : { ...current, norm: nextNorm }
                                                  );
                                                },
                                                mlpStepDomIdPrefix(blockIndex, componentIndex, stepIndex)
                                              )
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
  );
}
