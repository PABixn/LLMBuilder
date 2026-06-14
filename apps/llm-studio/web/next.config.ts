import type { NextConfig } from "next";

const isDesktopBuild = process.env.LLM_STUDIO_DESKTOP_BUILD === "1";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  output: isDesktopBuild ? "export" : undefined,
  trailingSlash: isDesktopBuild,
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
