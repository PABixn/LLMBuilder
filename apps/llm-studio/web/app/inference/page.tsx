"use client";

import { InferencePageView } from "./components/InferencePageView";
import { useInferenceController } from "./hooks/useInferenceController";

export default function InferencePage() {
  const controller = useInferenceController();
  return <InferencePageView controller={controller} />;
}
