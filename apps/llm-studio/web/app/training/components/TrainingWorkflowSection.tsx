import { formatStatusLabel } from "../lib/run";

export type TrainingWorkflowStep = {
  title: string;
  state: "ready" | "inProgress" | "waiting";
  status: string;
  actionLabel?: string;
  onAction?: () => void;
};

type TrainingWorkflowSectionProps = {
  steps: TrainingWorkflowStep[];
  startReady: boolean;
  launching: boolean;
  hasTrainingInProgress: boolean;
  onStartTraining: () => void;
};

export function TrainingWorkflowSection({
  steps,
  startReady,
  launching,
  hasTrainingInProgress,
  onStartTraining,
}: TrainingWorkflowSectionProps) {
  return (
    <section id="workflow" className="panelCard actionDeck">
      <div className="panelHead">
        <div>
          <p className="panelEyebrow">Workflow</p>
          <h2>Training steps</h2>
        </div>
      </div>
      <div className="workflowStepGrid" role="list" aria-label="Training launch steps">
        {steps.map((step, index) => (
          <article
            key={step.title}
            className={`workflowStepTile workflowStepTile-${
              step.state === "ready"
                ? "ready"
                : step.state === "inProgress"
                  ? "inProgress"
                  : "waiting"
            }`}
            role="listitem"
          >
            <p className="workflowStepTitle">{step.title}</p>
            <strong>{formatStatusLabel(step.status)}</strong>
            {step.onAction && step.actionLabel ? (
              <button
                type="button"
                className={`${
                  index === 3
                    ? "secondaryButton workflowStepAction workflowStepButtonCompact"
                    : "workflowStepLink workflowStepAction"
                }`}
                onClick={step.onAction}
              >
                {step.actionLabel}
              </button>
            ) : (
              <button
                type="button"
                className="primaryButton workflowStepAction"
                onClick={onStartTraining}
                disabled={!startReady}
              >
                {hasTrainingInProgress
                  ? "Training..."
                  : launching
                    ? "Starting..."
                    : "Start training"}
              </button>
            )}
          </article>
        ))}
      </div>
    </section>
  );
}
