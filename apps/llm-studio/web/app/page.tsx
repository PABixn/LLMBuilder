"use client";

import { StudioPageView } from "./studio/components/StudioPageView";
import { useStudioPageController } from "./studio/hooks/useStudioPageController";

export default function Page() {
  const pageProps = useStudioPageController();
  return <StudioPageView {...pageProps} />;
}
