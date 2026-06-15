import type { NextConfig } from "next";

import { isDesktopBuildEnvironment } from "./desktop-build-mode";

const isDesktopBuild = isDesktopBuildEnvironment(process.env);

const nextConfig: NextConfig = {
  reactStrictMode: true,
  output: isDesktopBuild ? "export" : undefined,
  trailingSlash: isDesktopBuild,
  images: {
    unoptimized: true,
  },
};

export default nextConfig;
