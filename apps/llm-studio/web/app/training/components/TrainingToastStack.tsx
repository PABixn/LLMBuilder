import type { ToastState } from "../types";

type TrainingToastStackProps = {
  toasts: ToastState[];
};

export function TrainingToastStack({ toasts }: TrainingToastStackProps) {
  return (
    <div className="trainingToastStack" aria-live="polite">
      {toasts.map((toast) => (
        <div
          key={toast.id}
          className={`trainingToast tone-${toast.level === "info" ? "success" : toast.level}`}
        >
          <div className="trainingToastTitle">{toast.title}</div>
          <div className="trainingToastBody">{toast.body}</div>
        </div>
      ))}
    </div>
  );
}
