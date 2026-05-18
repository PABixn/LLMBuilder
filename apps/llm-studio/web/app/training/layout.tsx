import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Training",
  description: "Train and monitor model runs.",
};

export default function TrainingLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return children;
}
