import type { Metadata } from "next";
import { THEME_BOOTSTRAP_SCRIPT } from "../lib/themeStorage";
import { RuntimeConfigGate } from "./shared/components/RuntimeConfigGate";
import "./globals.css";

export const metadata: Metadata = {
  title: "LLM Builder",
  description: "Build, train, and test local models.",
  icons: {
    icon: [
      { url: "/icons/icon-48x48.png", sizes: "48x48", type: "image/png" },
      { url: "/icons/icon-64x64.png", sizes: "64x64", type: "image/png" },
      { url: "/icons/icon-96x96.png", sizes: "96x96", type: "image/png" },
      { url: "/icons/icon-128x128.png", sizes: "128x128", type: "image/png" },
      { url: "/icons/icon-192x192.png", sizes: "192x192", type: "image/png" },
      { url: "/icons/icon-512x512.png", sizes: "512x512", type: "image/png" }
    ],
    shortcut: [{ url: "/icons/icon-64x64.png", type: "image/png" }],
    apple: [{ url: "/icons/icon-192x192.png", sizes: "192x192", type: "image/png" }]
  }
};

export default function RootLayout({
  children
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script
          dangerouslySetInnerHTML={{
            __html: THEME_BOOTSTRAP_SCRIPT,
          }}
        />
      </head>
      <body>
        <RuntimeConfigGate>{children}</RuntimeConfigGate>
      </body>
    </html>
  );
}
