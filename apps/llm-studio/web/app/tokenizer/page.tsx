"use client";

import { Suspense } from "react";

import { TokenizerPageContent } from "./components/TokenizerPageContent";

export default function TokenizerPage() {
  return (
    <Suspense fallback={null}>
      <TokenizerPageContent />
    </Suspense>
  );
}
