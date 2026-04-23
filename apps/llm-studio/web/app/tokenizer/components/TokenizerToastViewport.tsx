import type { ToastState } from "../types";

type TokenizerToastViewportProps = {
  toasts: ToastState[];
  onRemoveToast: (toastId: string) => void;
};

export function TokenizerToastViewport({
  toasts,
  onRemoveToast,
}: TokenizerToastViewportProps) {
  return (
    <aside className="toastViewport" aria-live="polite" aria-atomic="false">
      {toasts.map((toast) => (
        <div key={toast.id} className={`toast toast-${toast.level}`}>
          <div className="toastContent">
            <p>{toast.message}</p>
            <button
              type="button"
              className="toastClose"
              onClick={() => onRemoveToast(toast.id)}
              aria-label="Dismiss notification"
            >
              ×
            </button>
          </div>
          <div
            className="toastProgress"
            style={{ animationDuration: `${toast.durationMs}ms` }}
          />
        </div>
      ))}
    </aside>
  );
}
