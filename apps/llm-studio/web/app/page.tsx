"use client";

import { WorkspaceHomePageView } from "./home/components/WorkspaceHomePageView";
import { useWorkspaceHomeController } from "./home/hooks/useWorkspaceHomeController";

export default function WorkspaceHomePage() {
  const controller = useWorkspaceHomeController();
  return <WorkspaceHomePageView controller={controller} />;
}
