import type { ReactNode } from "react";
import { FiExternalLink } from "react-icons/fi";
import type { SimpleStepId, SimpleStepViewModel } from "../types";

const EXPERT_HREFS: Record<SimpleStepId, string> = {
  architecture: "/studio",
  tokenizer: "/tokenizer",
  training: "/training",
  inference: "/inference",
};

interface SimpleStepShellProps {
  activeStep: SimpleStepId;
  step: SimpleStepViewModel;
  children: ReactNode;
  onEdit: (step: SimpleStepId) => void;
}

export function SimpleStepShell({
  activeStep,
  step,
  children,
  onEdit,
}: SimpleStepShellProps) {
  const expanded = activeStep === step.id;

  return (
    <section
      className={`simpleStepShell is-${step.state} ${expanded ? "is-expanded" : "is-collapsed"}`}
      aria-labelledby={`${step.id}-title`}
    >
      <div className="simpleStepShellHeader">
        <div>
          <p className="simpleEyebrow">Step {step.index}</p>
          <h2 id={`${step.id}-title`}>{step.title}</h2>
          <p>{step.blocker ?? step.artifactLabel ?? step.actionLabel}</p>
        </div>
        <div className="simpleStepShellMeta">
          <span className={`simpleStatusPill is-${step.state}`}>{step.status}</span>
          {expanded ? (
            <a className="buttonGhost buttonSmall" href={EXPERT_HREFS[step.id]}>
              <FiExternalLink aria-hidden="true" /> Expert
            </a>
          ) : null}
          {!expanded ? (
            <button
              type="button"
              className="buttonGhost buttonSmall"
              onClick={() => onEdit(step.id)}
            >
              Edit
            </button>
          ) : null}
        </div>
      </div>
      {expanded ? <div className="simpleStepBody">{children}</div> : null}
    </section>
  );
}
