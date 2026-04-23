"use client";

import { Suspense } from "react";

import { TrainingPageContent } from "./components/TrainingPageContent";

export default function TrainingPage() {
  return (
    <Suspense fallback={null}>
      <TrainingPageContent />
    </Suspense>
  );
}
