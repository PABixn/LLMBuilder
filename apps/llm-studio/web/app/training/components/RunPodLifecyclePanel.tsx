import {
  FiCpu,
  FiRefreshCw,
  FiServer,
} from "react-icons/fi";

import type { TrainingJob } from "../../../lib/training/types";
import { formatDate } from "../../../lib/workspaceAssets";
import { formatStatusLabel } from "../lib/run";

interface RunPodLifecyclePanelProps {
  activeRun: TrainingJob;
}

export function RunPodLifecyclePanel({ activeRun }: RunPodLifecyclePanelProps) {
  if (activeRun.executor_kind !== "runpod_pod") {
    return null;
  }

  const lifecycleStatus = activeRun.executor_status ?? activeRun.stage;

  return (
    <div className="settingsStack">
      <div className="statusGrid">
        <div className="statusCard">
          <div className="statusCardIcon"><FiServer /></div>
          <div>
            <div className="statusCardTitle">RunPod lifecycle</div>
            <div className="statusCardValue">{formatStatusLabel(lifecycleStatus)}</div>
            <div className="statusCardDetail">Pod: {activeRun.runpod_pod_id ?? "Provisioning"}</div>
          </div>
        </div>
        <div className="statusCard">
          <div className="statusCardIcon"><FiCpu /></div>
          <div>
            <div className="statusCardTitle">Remote GPU</div>
            <div className="statusCardValue">{activeRun.runpod_gpu_type_id ?? "Selected"}</div>
            <div className="statusCardDetail">
              {activeRun.runpod_gpu_count} GPU · {formatStatusLabel(activeRun.runpod_cloud_type ?? "Cloud")} · {activeRun.runpod_data_center_id ?? "Any datacenter"}
            </div>
          </div>
        </div>
        <div className="statusCard">
          <div className="statusCardIcon"><FiRefreshCw /></div>
          <div>
            <div className="statusCardTitle">Sync</div>
            <div className="statusCardValue">{activeRun.runpod_last_sync_at ? formatDate(activeRun.runpod_last_sync_at) : "Waiting"}</div>
            <div className="statusCardDetail">{activeRun.remote_error ?? "No remote errors"}</div>
          </div>
        </div>
      </div>
    </div>
  );
}
