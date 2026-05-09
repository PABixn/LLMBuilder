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
            <div className="statusCardDetail">{activeRun.remote_error ?? "No remote errors"}</div>
          </div>
        </div>
      </div>
    </div>
  );
}
