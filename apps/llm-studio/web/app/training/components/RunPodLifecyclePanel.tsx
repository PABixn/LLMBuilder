import {
  FiCpu,
  FiRefreshCw,
  FiServer,
} from "react-icons/fi";

import type { TrainingJob } from "../../../lib/training/types";
import { formatDate } from "../../../lib/workspaceAssets";

interface RunPodLifecyclePanelProps {
  activeRun: TrainingJob;
}

export function RunPodLifecyclePanel({ activeRun }: RunPodLifecyclePanelProps) {
  if (activeRun.executor_kind !== "runpod_pod") {
    return null;
  }
  const cleanupPod = String(activeRun.runpod_cleanup_policy?.pod ?? "delete_after_sync");
  const cleanupLabel =
    cleanupPod === "delete_after_sync"
      ? "delete pod after sync"
      : cleanupPod === "stop_after_sync"
        ? "stop pod after sync"
        : "keep pod running";

  return (
    <div className="settingsStack">
      <div className="statusGrid">
        <div className="statusCard">
          <div className="statusCardIcon"><FiServer /></div>
          <div>
            <div className="statusCardTitle">RunPod lifecycle</div>
            <div className="statusCardValue">{activeRun.executor_status ?? activeRun.stage}</div>
            <div className="statusCardDetail">Pod: {activeRun.runpod_pod_id ?? "provisioning"}</div>
          </div>
        </div>
        <div className="statusCard">
          <div className="statusCardIcon"><FiCpu /></div>
          <div>
            <div className="statusCardTitle">Remote GPU</div>
            <div className="statusCardValue">{activeRun.runpod_gpu_type_id ?? "selected"}</div>
            <div className="statusCardDetail">
              {activeRun.runpod_gpu_count} GPU · {activeRun.runpod_cloud_type ?? "cloud"} · {activeRun.runpod_data_center_id ?? "any datacenter"}
            </div>
          </div>
        </div>
        <div className="statusCard">
          <div className="statusCardIcon"><FiRefreshCw /></div>
          <div>
            <div className="statusCardTitle">Sync</div>
            <div className="statusCardValue">{activeRun.runpod_last_sync_at ? formatDate(activeRun.runpod_last_sync_at) : "waiting"}</div>
            <div className="statusCardDetail">{activeRun.remote_error ?? "Logs, metrics, samples, checkpoints, and the final manifest sync to this machine."}</div>
          </div>
        </div>
      </div>
      <div className="trainingWorkflowList">
        <div className="trainingWorkflowStep trainingWorkflowStep-ready">
          <span className="trainingWorkflowStepStatus">Pod</span>
          <span>{activeRun.runpod_pod_id ?? "Provisioning"}</span>
        </div>
        {activeRun.runpod_agent_base_url ? (
          <div className="trainingWorkflowStep trainingWorkflowStep-ready">
            <span className="trainingWorkflowStepStatus">Agent</span>
            <span>{activeRun.runpod_agent_base_url}</span>
          </div>
        ) : null}
        <div className="trainingWorkflowStep trainingWorkflowStep-ready">
          <span className="trainingWorkflowStepStatus">Heartbeat</span>
          <span>{activeRun.runpod_last_heartbeat_at ? formatDate(activeRun.runpod_last_heartbeat_at) : "Waiting for pod agent"}</span>
        </div>
        <div className="trainingWorkflowStep trainingWorkflowStep-ready">
          <span className="trainingWorkflowStepStatus">Cleanup</span>
          <span>{cleanupLabel}</span>
        </div>
        <div className="trainingWorkflowStep trainingWorkflowStep-waiting">
          <span className="trainingWorkflowStepStatus">Recovery</span>
          <span>Manual resync, cleanup, and reattach actions are unavailable after an app restart.</span>
        </div>
      </div>
    </div>
  );
}
