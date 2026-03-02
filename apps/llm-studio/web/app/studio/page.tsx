"use client";

import { useStudioPageController } from "./hooks/useStudioPageController";
import { StudioPageView } from "./components/StudioPageView";

export default function StudioPage() {
  const controllerProps = useStudioPageController();
  
  return (
    <StudioPageView {...controllerProps} />
  );
}
