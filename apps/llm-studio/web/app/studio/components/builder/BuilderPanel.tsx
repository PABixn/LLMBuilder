import type { DragEvent, ReactNode } from "react";
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

import { ACTIVATION_TYPES, type ActivationType, type NormConfig } from "../../../../lib/defaults";

import { DropSlot, PaletteTile } from "../primitives";
import type {
  BuilderMetrics,
  ConsecutiveBlockGroup,
  MlpStepKind,
  StudioBlock,
  StudioComponent,
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
  beginDragBlock: (event: DragEvent<HTMLDivElement>, blockId: string) => void;
  beginDragComponent: (
    event: DragEvent<HTMLDivElement>,
    fromBlockId: string,
    componentId: string
  ) => void;
  beginDragPaletteMlpStep: (event: DragEvent<HTMLDivElement>, stepKind: MlpStepKind) => void;
  beginDragMlpStep: (
    event: DragEvent<HTMLDivElement>,
    fromBlockId: string,
    fromComponentId: string,
    stepId: string
  ) => void;
  clearDragState: () => void;
  markDropTarget: (event: DragEvent<HTMLDivElement>, key: string) => void;
  handleDropBlock: (event: DragEvent<HTMLDivElement>, insertIndex: number) => void;
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
  updateComponent,
  updateMlpStep,
  beginDragBlock,
  beginDragComponent,
  beginDragPaletteMlpStep,
  beginDragMlpStep,
  clearDragState,
  markDropTarget,
  handleDropBlock,
  handleDropComponent,
  handleDropMlpStep,
}: BuilderPanelProps) {
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
            <span>learnable_gamma</span>
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
            <div
              className="dragBadge"
              draggable
              onDragStart={(event) => beginDragBlock(event, block.id)}
              onDragEnd={clearDragState}
              title={`Drag block ${blockIndex + 1}`}
              aria-hidden
            >
              <FiMove />
            </div>
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
          <DropSlot
            compact
            active={dragOverKey === `component-slot-${block.id}-0`}
            label="Insert"
            onDragOver={(event) => markDropTarget(event, `component-slot-${block.id}-0`)}
            onDrop={(event) => handleDropComponent(event, block.id, 0)}
          />

          {block.components.length === 0 ? (
            <div className="emptyLaneHint">Drop a palette component here to start this block.</div>
          ) : null}

          {block.components.map((component, componentIndex) => (
            <Fragment key={component.id}>
              <section
                className={`componentCard kind-${component.kind}${expandedComponentIds.has(component.id) ? "" : " isCollapsed"}`}
              >
                <div className="componentCardHead">
                  <div
                    className="dragBadge"
                    draggable
                    onDragStart={(event) => beginDragComponent(event, block.id, component.id)}
                    onDragEnd={clearDragState}
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
                    <button
                      type="button"
                      className="iconButton"
                      onClick={() => toggleExpandedComponent(component.id)}
                      aria-label={`${expandedComponentIds.has(component.id) ? "Collapse" : "Expand"} ${labelForComponentKind(component.kind)} component settings`}
                      title={expandedComponentIds.has(component.id) ? "Collapse settings" : "Expand settings"}
                      aria-expanded={expandedComponentIds.has(component.id)}
                    >
                      {expandedComponentIds.has(component.id) ? <FiChevronDown /> : <FiChevronRight />}
                    </button>
                    <button
                      type="button"
                      className="iconButton danger"
                      onClick={() => removeComponent(block.id, component.id)}
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
                        <div className="fieldGrid compact">
                          <label
                            className="fieldLabel"
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

                        <div className="mlpPaletteRow">
                          <div className="miniLabel">MLP Step Palette</div>
                          <div className="miniPaletteGrid">
                            {([
                              {
                                kind: "linear",
                                subtitle: "Linear layer",
                                colorClass: "tone-linear",
                              },
                              {
                                kind: "norm",
                                subtitle: "Norm step",
                                colorClass: "tone-norm",
                              },
                              {
                                kind: "activation",
                                subtitle: "Activation step",
                                colorClass: "tone-activation",
                              },
                            ] as const).map((entry) => (
                              <PaletteTile
                                key={`${component.id}-${entry.kind}`}
                                title={labelForMlpStepKind(entry.kind)}
                                subtitle={entry.subtitle}
                                colorClass={entry.colorClass}
                                draggable
                                onDragStart={(event) => beginDragPaletteMlpStep(event, entry.kind)}
                                onDragEnd={clearDragState}
                              />
                            ))}
                          </div>
                        </div>

                        <div className="mlpSequenceShell">
                          <div className="miniLabel">Sequence</div>
                          <div className="mlpSequenceList">
                            <DropSlot
                              compact
                              active={dragOverKey === `mlp-slot-${block.id}-${component.id}-0`}
                              label="Insert step"
                              onDragOver={(event) =>
                                markDropTarget(event, `mlp-slot-${block.id}-${component.id}-0`)
                              }
                              onDrop={(event) => handleDropMlpStep(event, block.id, component.id, 0)}
                            />
                            {component.mlp.sequence.length === 0 ? (
                              <div className="emptyLaneHint compact">
                                Drop a linear/norm/activation step here.
                              </div>
                            ) : null}
                            {component.mlp.sequence.map((step, stepIndex) => (
                              <Fragment key={step.id}>
                                <div
                                  className={`mlpStepCard kind-${step.kind}${expandedMlpStepIds.has(step.id) ? "" : " isCollapsed"}`}
                                >
                                  <div className="componentCardHead">
                                    <div
                                      className="dragBadge"
                                      draggable
                                      onDragStart={(event) =>
                                        beginDragMlpStep(event, block.id, component.id, step.id)
                                      }
                                      onDragEnd={clearDragState}
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
                                      <button
                                        type="button"
                                        className="iconButton"
                                        onClick={() => toggleExpandedMlpStep(step.id)}
                                        aria-label={`${expandedMlpStepIds.has(step.id) ? "Collapse" : "Expand"} MLP step settings`}
                                        title={
                                          expandedMlpStepIds.has(step.id)
                                            ? "Collapse settings"
                                            : "Expand settings"
                                        }
                                        aria-expanded={expandedMlpStepIds.has(step.id)}
                                      >
                                        {expandedMlpStepIds.has(step.id) ? (
                                          <FiChevronDown />
                                        ) : (
                                          <FiChevronRight />
                                        )}
                                      </button>
                                      <button
                                        type="button"
                                        className="iconButton danger"
                                        onClick={() => removeMlpStep(block.id, component.id, step.id)}
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
                                <DropSlot
                                  compact
                                  active={
                                    dragOverKey === `mlp-slot-${block.id}-${component.id}-${stepIndex + 1}`
                                  }
                                  label="Insert step"
                                  onDragOver={(event) =>
                                    markDropTarget(
                                      event,
                                      `mlp-slot-${block.id}-${component.id}-${stepIndex + 1}`
                                    )
                                  }
                                  onDrop={(event) =>
                                    handleDropMlpStep(event, block.id, component.id, stepIndex + 1)
                                  }
                                />
                              </Fragment>
                            ))}
                          </div>
                        </div>
                      </div>
                    ) : null}
                  </div>
                ) : null}
              </section>
              <DropSlot
                compact
                active={dragOverKey === `component-slot-${block.id}-${componentIndex + 1}`}
                label="Insert"
                onDragOver={(event) =>
                  markDropTarget(event, `component-slot-${block.id}-${componentIndex + 1}`)
                }
                onDrop={(event) => handleDropComponent(event, block.id, componentIndex + 1)}
              />
            </Fragment>
          ))}
        </div>
      </article>
    );
  }

  return (
      <section id="block-builder" className="panelCard builderPanel">
        <div className="panelHead">
          <div>
            <p className="panelEyebrow">Visual Builder</p>
            <h2>Horizontal block canvas</h2>
            <p className="panelCopy">
              Scroll horizontally for model depth and vertically for details. Drag blocks to reorder, then expand only the nodes you want to edit.
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
          <DropSlot
            active={dragOverKey === "block-slot-0"}
            label="Insert block"
            onDragOver={(event) => markDropTarget(event, "block-slot-0")}
            onDrop={(event) => handleDropBlock(event, 0)}
          />

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
                    <div className="blockGroupHead">
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
                      <button
                        type="button"
                        className="iconButton"
                        onClick={() => toggleExpandedBlockGroup(group.key)}
                        aria-label={groupExpanded ? "Collapse identical block group" : "Expand identical block group"}
                        title={groupExpanded ? "Collapse identical block group" : "Expand identical block group"}
                        aria-expanded={groupExpanded}
                      >
                        {groupExpanded ? <FiChevronDown /> : <FiChevronRight />}
                      </button>
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
                                <DropSlot
                                  active={dragOverKey === `block-slot-${absoluteIndex + 1}`}
                                  label="Insert block"
                                  onDragOver={(event) =>
                                    markDropTarget(event, `block-slot-${absoluteIndex + 1}`)
                                  }
                                  onDrop={(event) => handleDropBlock(event, absoluteIndex + 1)}
                                />
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
                <DropSlot
                  active={dragOverKey === `block-slot-${group.endIndex + 1}`}
                  label="Insert block"
                  onDragOver={(event) => markDropTarget(event, `block-slot-${group.endIndex + 1}`)}
                  onDrop={(event) => handleDropBlock(event, group.endIndex + 1)}
                />
              </Fragment>
            );
          })}
          </div>
        </div>
      </section>
  );
}
