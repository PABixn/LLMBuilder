"use client";

import { useCallback, useEffect, useState, type ReactNode } from "react";

import {
  type DesktopCloseRequest,
  cancelDesktopRuntimeStart,
  exportDesktopDiagnostics,
  listenForDesktopCloseRequest,
  listenForDesktopRetryRequest,
  listenForDesktopStartupProgress,
  openDesktopData,
  openDesktopLogs,
  quitDesktopApp,
  stopAndExitDesktopApp,
} from "../../../lib/desktopBridge";
import { initializeRuntimeConfig } from "../../../lib/runtimeConfig";

type RuntimeState =
  | { status: "loading"; message: string }
  | { status: "ready" }
  | { status: "failed"; message: string };

export function RuntimeConfigGate({ children }: { children: ReactNode }) {
  const [state, setState] = useState<RuntimeState>({
    status: "loading",
    message: "Connecting to the local runtime...",
  });
  const [closeRequest, setCloseRequest] = useState<DesktopCloseRequest | null>(null);

  const runDesktopAction = useCallback(async (action: () => Promise<unknown>) => {
    try {
      await action();
    } catch (error) {
      setState({
        status: "failed",
        message: error instanceof Error ? error.message : "The desktop action failed.",
      });
    }
  }, []);

  const connect = useCallback(async (retry = false) => {
    setState({
      status: "loading",
      message: retry ? "Retrying the local runtime..." : "Connecting to the local runtime...",
    });
    try {
      await initializeRuntimeConfig({ retry });
      setState({ status: "ready" });
    } catch (error) {
      setState({
        status: "failed",
        message: error instanceof Error ? error.message : "The local runtime failed to start.",
      });
    }
  }, []);

  useEffect(() => {
    void connect();
  }, [connect]);

  useEffect(() => {
    let unlisten: () => void = () => undefined;
    void listenForDesktopStartupProgress((progress) => {
      setState((current) =>
        current.status === "loading"
          ? { status: "loading", message: progress.message }
          : current
      );
    }).then((cleanup) => {
      unlisten = cleanup;
    });
    return () => unlisten();
  }, []);

  useEffect(() => {
    let unlisten: () => void = () => undefined;
    void listenForDesktopCloseRequest(setCloseRequest).then((cleanup) => {
      unlisten = cleanup;
    });
    return () => unlisten();
  }, []);

  useEffect(() => {
    let unlisten: () => void = () => undefined;
    void listenForDesktopRetryRequest(() => {
      void connect(true);
    }).then((cleanup) => {
      unlisten = cleanup;
    });
    return () => unlisten();
  }, [connect]);

  if (state.status === "ready") {
    return (
      <>
        {children}
        {closeRequest ? (
          <div className="runtimeCloseBackdrop" role="presentation">
            <section
              className="panelCard runtimeCloseDialog"
              role="alertdialog"
              aria-modal="true"
              aria-labelledby="runtime-close-title"
              aria-describedby="runtime-close-description"
            >
              <h2 id="runtime-close-title">Active work is still running</h2>
              <p id="runtime-close-description" className="panelCopy">
                {closeRequest.message}
              </p>
              <div className="actionCluster runtimeGateActions">
                <button
                  type="button"
                  className="buttonPrimary"
                  onClick={() => setCloseRequest(null)}
                >
                  Return to app
                </button>
                <button
                  type="button"
                  className="buttonDanger"
                  onClick={() => void runDesktopAction(stopAndExitDesktopApp)}
                >
                  {closeRequest.active_jobs.has_active_runpod_training
                    ? "Exit; RunPod may keep billing"
                    : "Stop local work and exit"}
                </button>
              </div>
            </section>
          </div>
        ) : null}
      </>
    );
  }

  return (
    <main className="runtimeGate">
      <section className="panelCard runtimeGateCard" aria-live="polite">
        <h1>{state.status === "failed" ? "LLM Studio could not start." : "Starting LLM Studio..."}</h1>
        <p className="panelCopy">{state.message}</p>
        {state.status === "failed" ? (
          <div className="actionCluster runtimeGateActions">
            <button type="button" className="buttonPrimary" onClick={() => void connect(true)}>
              Retry backend
            </button>
            <button type="button" className="buttonGhost" onClick={() => void runDesktopAction(openDesktopLogs)}>
              Open logs
            </button>
            <button type="button" className="buttonGhost" onClick={() => void runDesktopAction(openDesktopData)}>
              Open data folder
            </button>
            <button type="button" className="buttonGhost" onClick={() => void runDesktopAction(exportDesktopDiagnostics)}>
              Export diagnostics
            </button>
            <button type="button" className="buttonDanger" onClick={() => void runDesktopAction(quitDesktopApp)}>
              Quit
            </button>
          </div>
        ) : (
          <>
            <div className="runtimeGateProgress" aria-hidden="true" />
            <div className="actionCluster runtimeGateActions">
              <button
                type="button"
                className="buttonGhost"
                onClick={() => void runDesktopAction(cancelDesktopRuntimeStart)}
              >
                Cancel startup
              </button>
            </div>
          </>
        )}
      </section>
    </main>
  );
}
