import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Inference Studio",
  description: "Run autocomplete inference against completed LLM training artifacts.",
};

export default function InferenceLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return children;
}
