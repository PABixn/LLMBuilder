import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "LLM Training",
  description:
    "Launch, validate, and monitor end-to-end LLM training runs with persisted workspace artifacts.",
};

export default function TrainingLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return children;
}
