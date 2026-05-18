import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Inference",
  description: "Generate text with a trained model.",
};

export default function InferenceLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return children;
}
