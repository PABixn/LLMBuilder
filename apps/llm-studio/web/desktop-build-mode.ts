type DesktopBuildEnvironment = Readonly<Record<string, string | undefined>>;

export function isDesktopBuildEnvironment(environment: DesktopBuildEnvironment): boolean {
  return (
    environment.LLM_STUDIO_DESKTOP_BUILD === "1" ||
    environment.npm_lifecycle_event === "build:desktop"
  );
}
