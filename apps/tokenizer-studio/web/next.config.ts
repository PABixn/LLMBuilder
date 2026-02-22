import type { NextConfig } from "next";

const isDesktopBuild = process.env.TOKENIZER_STUDIO_DESKTOP_BUILD === "1";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  output: isDesktopBuild ? "export" : undefined,
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
