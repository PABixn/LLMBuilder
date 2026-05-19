import type { SimpleStepId, SimpleStepViewModel } from "../types";

interface SimpleStepperProps {
  activeStep: SimpleStepId;
  steps: SimpleStepViewModel[];
  onStepSelect: (step: SimpleStepId) => void;
}

export function SimpleStepper({
  activeStep,
  steps,
  onStepSelect,
}: SimpleStepperProps) {
  return (
    <section className="simpleStepper" aria-label="Simple Mode steps">
      {steps.map((step) => (
        <button
          key={step.id}
          type="button"
          className={`simpleStepTab is-${step.state}`}
          aria-current={activeStep === step.id ? "step" : undefined}
          onClick={() => onStepSelect(step.id)}
        >
          <span className="simpleStepIndex">{step.index}</span>
          <span className="simpleStepTabBody">
            <span className="simpleStepTabTitle">{step.title}</span>
            <span className="simpleStepTabStatus">{step.status}</span>
          </span>
        </button>
      ))}
    </section>
  );
}
