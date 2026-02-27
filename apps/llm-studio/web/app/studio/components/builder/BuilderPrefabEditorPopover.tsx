import { useEffect, useState, type ReactNode, type RefObject } from "react";
import { createPortal } from "react-dom";
import {
  FiChevronDown,
  FiChevronUp,
  FiPlus,
  FiTrash2,
  FiX,
} from "react-icons/fi";

import { ACTIVATION_TYPES, type ActivationType, type BlockComponent } from "../../../../lib/defaults";

import type { StudioComponent, StudioComponentPrefab, StudioMlpStep } from "../../types";
import {
  createDefaultStudioMlpStep,
  labelForActivationType,
  labelForComponentKind,
  labelForMlpStepKind,
  studioComponentFromConfig,
  studioComponentToConfig,
} from "../../utils/document";
import {
  integerInputValue,
  numberInputValue,
  parseIntegerInput,
  parseNumberInput,
} from "../../utils/format";
import { renderNormFields } from "./BuilderControls";

type BuilderPrefabEditorPopoverProps = {
  prefab: StudioComponentPrefab | null;
  popoverRef: RefObject<HTMLDivElement | null>;
  position: { left: number; top: number } | null;
  closeEditor: () => void;
  updateComponentPrefab: (
    prefabId: string,
    nextName: string,
    nextComponent: StudioComponent
  ) => string | null;
  replaceAllComponentsWithComponentSettings: (
    prefabName: string,
    kind: StudioComponent["kind"],
    component: BlockComponent
  ) => void;
  deleteComponentPrefab: (prefabId: string) => void;
};

export function BuilderPrefabEditorPopover({
  prefab,
  popoverRef,
  position,
  closeEditor,
  updateComponentPrefab,
  replaceAllComponentsWithComponentSettings,
  deleteComponentPrefab,
}: BuilderPrefabEditorPopoverProps) {
  const [nameDraft, setNameDraft] = useState<string>("");
  const [nameError, setNameError] = useState<string | null>(null);
  const [componentDraft, setComponentDraft] = useState<StudioComponent | null>(null);

  useEffect(() => {
    if (!prefab) {
      return;
    }
    setNameDraft(prefab.name);
    setComponentDraft(studioComponentFromConfig(prefab.component));
    setNameError(null);
  }, [prefab]);

  if (!prefab || !componentDraft || typeof document === "undefined") {
    return null;
  }
  const activePrefab = prefab;
  const activeComponentDraft = componentDraft;

  function updateDraft(updater: (current: StudioComponent) => StudioComponent): void {
    setComponentDraft((current) => (current ? updater(current) : current));
  }

  function updateMlpDraftStep(stepId: string, updater: (step: StudioMlpStep) => StudioMlpStep): void {
    updateDraft((current) => {
      if (current.kind !== "mlp") {
        return current;
      }
      return {
        ...current,
        mlp: {
          ...current.mlp,
          sequence: current.mlp.sequence.map((step) => (step.id === stepId ? updater(step) : step)),
        },
      };
    });
  }

  function removeMlpDraftStep(stepId: string): void {
    updateDraft((current) => {
      if (current.kind !== "mlp") {
        return current;
      }
      return {
        ...current,
        mlp: {
          ...current.mlp,
          sequence: current.mlp.sequence.filter((step) => step.id !== stepId),
        },
      };
    });
  }

  function moveMlpDraftStep(stepIndex: number, direction: -1 | 1): void {
    updateDraft((current) => {
      if (current.kind !== "mlp") {
        return current;
      }
      const nextIndex = stepIndex + direction;
      if (nextIndex < 0 || nextIndex >= current.mlp.sequence.length) {
        return current;
      }
      const nextSequence = current.mlp.sequence.slice();
      const [moved] = nextSequence.splice(stepIndex, 1);
      nextSequence.splice(nextIndex, 0, moved);
      return {
        ...current,
        mlp: {
          ...current.mlp,
          sequence: nextSequence,
        },
      };
    });
  }

  function insertMlpDraftStep(kind: "linear" | "norm" | "activation"): void {
    updateDraft((current) => {
      if (current.kind !== "mlp") {
        return current;
      }
      return {
        ...current,
        mlp: {
          ...current.mlp,
          sequence: [...current.mlp.sequence, createDefaultStudioMlpStep(kind)],
        },
      };
    });
  }

  function saveDraft(replaceAll: boolean): void {
    const normalizedName = nameDraft.trim();
    if (!normalizedName) {
      setNameError("Name is required.");
      return;
    }

    const savedName = updateComponentPrefab(activePrefab.id, normalizedName, activeComponentDraft);
    if (!savedName) {
      return;
    }

    if (replaceAll) {
      replaceAllComponentsWithComponentSettings(
        savedName,
        activeComponentDraft.kind,
        studioComponentToConfig(activeComponentDraft)
      );
    }

    closeEditor();
  }

  function renderComponentDesigner(): ReactNode {
    const idPrefix = `prefab-editor-${activePrefab.id}`;

    if (activeComponentDraft.kind === "attention") {
      return (
        <div className="fieldGrid compact">
          <label className="fieldLabel" htmlFor={`${idPrefix}-heads`}>
            <span>Heads</span>
            <input
              id={`${idPrefix}-heads`}
              type="number"
              min={1}
              step={1}
              value={integerInputValue(activeComponentDraft.attention.n_head)}
              onChange={(event) =>
                updateDraft((current) =>
                  current.kind !== "attention"
                    ? current
                    : {
                        ...current,
                        attention: {
                          ...current.attention,
                          n_head: parseIntegerInput(event.target.value, current.attention.n_head),
                        },
                      }
                )
              }
            />
          </label>
          <label className="fieldLabel" htmlFor={`${idPrefix}-kv-heads`}>
            <span>KV heads</span>
            <input
              id={`${idPrefix}-kv-heads`}
              type="number"
              min={1}
              step={1}
              value={integerInputValue(activeComponentDraft.attention.n_kv_head)}
              onChange={(event) =>
                updateDraft((current) =>
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
      );
    }

    if (activeComponentDraft.kind === "norm") {
      return (
        <>
          {renderNormFields(
            activeComponentDraft.norm,
            (next) => updateDraft((current) => (current.kind !== "norm" ? current : { ...current, norm: next })),
            idPrefix
          )}
        </>
      );
    }

    if (activeComponentDraft.kind === "activation") {
      return (
        <div className="fieldGrid compact">
          <label className="fieldLabel" htmlFor={`${idPrefix}-activation`}>
            <span>Activation</span>
            <select
              id={`${idPrefix}-activation`}
              value={activeComponentDraft.activation.type}
              onChange={(event) =>
                updateDraft((current) =>
                  current.kind !== "activation"
                    ? current
                    : {
                        ...current,
                        activation: { type: event.target.value as ActivationType },
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
      );
    }

    return (
      <div className="prefabMlpDesigner">
        <div className="fieldGrid compact prefabMlpFields">
          <label className="fieldLabel" htmlFor={`${idPrefix}-multiplier`}>
            <span>Multiplier</span>
            <input
              id={`${idPrefix}-multiplier`}
              type="number"
              min={0.001}
              step="any"
              value={numberInputValue(activeComponentDraft.mlp.multiplier)}
              onChange={(event) =>
                updateDraft((current) =>
                  current.kind !== "mlp"
                    ? current
                    : {
                        ...current,
                        mlp: {
                          ...current.mlp,
                          multiplier: parseNumberInput(event.target.value, current.mlp.multiplier),
                        },
                      }
                )
              }
            />
          </label>
        </div>

        <div className="prefabStepList" role="list" aria-label="Prefab MLP steps">
          {activeComponentDraft.mlp.sequence.length === 0 ? (
            <p className="prefabMiniHint" role="listitem">
              No MLP steps. Add one below.
            </p>
          ) : null}
          {activeComponentDraft.mlp.sequence.map((step: StudioMlpStep, stepIndex: number) => (
            <div key={step.id} className="prefabStepCard" role="listitem">
              <div className="prefabStepHead">
                <span className={`flowMiniChip kind-${step.kind}`}>
                  {stepIndex + 1}. {labelForMlpStepKind(step.kind)}
                </span>
                <div className="prefabStepActions">
                  <button
                    type="button"
                    className="iconButton"
                    onClick={() => moveMlpDraftStep(stepIndex, -1)}
                    aria-label="Move step up"
                    title="Move step up"
                    disabled={stepIndex === 0}
                  >
                    <FiChevronUp />
                  </button>
                  <button
                    type="button"
                    className="iconButton"
                    onClick={() => moveMlpDraftStep(stepIndex, 1)}
                    aria-label="Move step down"
                    title="Move step down"
                    disabled={stepIndex === activeComponentDraft.mlp.sequence.length - 1}
                  >
                    <FiChevronDown />
                  </button>
                  <button
                    type="button"
                    className="iconButton danger"
                    onClick={() => removeMlpDraftStep(step.id)}
                    aria-label="Delete step"
                    title="Delete step"
                  >
                    <FiTrash2 />
                  </button>
                </div>
              </div>

              {step.kind === "linear" ? (
                <label className="toggleField" htmlFor={`${idPrefix}-${step.id}-bias`}>
                  <input
                    id={`${idPrefix}-${step.id}-bias`}
                    type="checkbox"
                    checked={step.linear.bias}
                    onChange={(event) =>
                      updateMlpDraftStep(step.id, (current) =>
                        current.kind !== "linear"
                          ? current
                          : { ...current, linear: { bias: event.target.checked } }
                      )
                    }
                  />
                  <span>Bias</span>
                </label>
              ) : null}

              {step.kind === "norm" ? (
                <>
                  {renderNormFields(
                    step.norm,
                    (next) =>
                      updateMlpDraftStep(step.id, (current) =>
                        current.kind !== "norm" ? current : { ...current, norm: next }
                      ),
                    `${idPrefix}-${step.id}`
                  )}
                </>
              ) : null}

              {step.kind === "activation" ? (
                <div className="fieldGrid compact">
                  <label className="fieldLabel" htmlFor={`${idPrefix}-${step.id}-activation`}>
                    <span>Activation</span>
                    <select
                      id={`${idPrefix}-${step.id}-activation`}
                      value={step.activation.type}
                      onChange={(event) =>
                        updateMlpDraftStep(step.id, (current) =>
                          current.kind !== "activation"
                            ? current
                            : {
                                ...current,
                                activation: { type: event.target.value as ActivationType },
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
          ))}
        </div>

        <div className="prefabStepInsertButtons">
          {(["linear", "norm", "activation"] as const).map((kind) => (
            <button
              key={kind}
              type="button"
              className="buttonGhost prefabMiniButton"
              onClick={() => insertMlpDraftStep(kind)}
            >
              <FiPlus /> {labelForMlpStepKind(kind)}
            </button>
          ))}
        </div>
      </div>
    );
  }

  return createPortal(
    <div
      ref={popoverRef}
      className="prefabEditorPopover"
      data-prefab-editor-root
      role="dialog"
      aria-label="Edit prefab"
      style={position ? { left: position.left, top: position.top } : { left: -9999, top: -9999 }}
    >
      <div className="prefabEditorHead">
        <div>
          <h4>{labelForComponentKind(activePrefab.kind)} prefab</h4>
          <p>Mini designer</p>
        </div>
        <button
          type="button"
          className="iconButton"
          onClick={closeEditor}
          aria-label="Close prefab editor"
          title="Close"
        >
          <FiX />
        </button>
      </div>

      <label className="fieldLabel prefabNameField" htmlFor={`prefab-name-${activePrefab.id}`}>
        <span>Name</span>
        <input
          id={`prefab-name-${activePrefab.id}`}
          type="text"
          value={nameDraft}
          onChange={(event) => {
            setNameDraft(event.target.value);
            if (nameError) {
              setNameError(null);
            }
          }}
          placeholder="Prefab name"
          maxLength={64}
        />
      </label>
      {nameError ? <p className="prefabFieldError">{nameError}</p> : null}

      <div className="prefabMiniDesignerShell">{renderComponentDesigner()}</div>

      <div className="prefabEditorActions">
        <button type="button" className="buttonGhost" onClick={closeEditor}>
          Cancel
        </button>
        <button
          type="button"
          className="buttonGhost"
          onClick={() => saveDraft(true)}
          title={`Replace all ${labelForComponentKind(activePrefab.kind)} components`}
        >
          Save + replace all
        </button>
        <button type="button" className="buttonGhost prefabSaveButton" onClick={() => saveDraft(false)}>
          Save changes
        </button>
        <button
          type="button"
          className="buttonGhost prefabDeleteButton"
          onClick={() => {
            deleteComponentPrefab(activePrefab.id);
            closeEditor();
          }}
        >
          <FiTrash2 /> Delete
        </button>
      </div>
    </div>,
    document.body
  );
}
