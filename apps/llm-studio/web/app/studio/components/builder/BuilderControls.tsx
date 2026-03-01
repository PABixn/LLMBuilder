import type { KeyboardEvent as ReactKeyboardEvent, ReactNode } from "react";

import { ACTIVATION_TYPES, type ActivationType, type NormConfig } from "../../../../lib/defaults";

import type { StudioComponent, StudioMlpStep } from "../../types";
import {
  labelForActivationType,
  labelForComponentKind,
  labelForNormType,
  summarizeComponent,
} from "../../utils/document";

export function handleToggleKeyDown(
  event: ReactKeyboardEvent<HTMLElement>,
  toggle: () => void
): void {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    toggle();
  }
}

export function renderNormFields(
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

export function renderInlineNormControls(
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

export function renderInlineActivationControls(
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

export function renderInlineLinearControls(
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

export function renderQuickMapChips(
  components: StudioComponent[],
  chipKeyPrefix: string,
  quickMapChipLimit = 4
): ReactNode {
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

export function isInlineSimpleComponent(component: StudioComponent): boolean {
  return component.kind === "activation";
}

export function isInlineSimpleMlpStep(_step: StudioMlpStep): boolean {
  return true;
}
