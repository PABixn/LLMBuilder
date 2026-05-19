"use client";

import { SimpleModePageView } from "./components/SimpleModePageView";
import { useSimpleModeController } from "./hooks/useSimpleModeController";

export default function SimpleModePage() {
  const controller = useSimpleModeController();
  return <SimpleModePageView controller={controller} />;
}
