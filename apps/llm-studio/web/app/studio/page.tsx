"use client";

import { Suspense } from "react";

import { useStudioPageController } from "./hooks/useStudioPageController";
import { StudioPageView } from "./components/StudioPageView";

function StudioPageContent() {
  const controllerProps = useStudioPageController();
  
  return (
    <StudioPageView {...controllerProps} />
  );
}

export default function StudioPage() {
  return (
    <Suspense fallback={null}>
      <StudioPageContent />
    </Suspense>
  );
}
