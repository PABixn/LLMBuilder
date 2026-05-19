"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect } from "react";
import { FiMoon, FiSun } from "react-icons/fi";

import { useThemeMode } from "../../../lib/theme";
import { useUiMode } from "../hooks/useUiMode";
import {
  EXPERT_NAV_LINKS,
  SIMPLE_NAV_LINKS,
  expertRouteForSimpleStep,
  isRouteActive,
  type SimpleStepId,
} from "../lib/navigation";

type AppTopNavProps = {
  activeSimpleStep?: SimpleStepId | null;
};

export function AppTopNav({ activeSimpleStep = null }: AppTopNavProps) {
  const pathname = usePathname() ?? "/";
  const router = useRouter();
  const [theme, setTheme] = useThemeMode();
  const [uiMode, setUiMode] = useUiMode();
  const isSimpleRoute = pathname === "/simple" || pathname.startsWith("/simple/");
  const links = uiMode === "simple" ? SIMPLE_NAV_LINKS : EXPERT_NAV_LINKS;
  const modeSwitchLabel = uiMode === "simple" ? "Expert Mode" : "Simple Mode";
  const modeSwitchTitle =
    uiMode === "simple" ? "Switch to Expert Mode" : "Switch to Simple Mode";

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    const requestedMode = new URLSearchParams(window.location.search).get("mode");
    if (requestedMode === "simple") {
      setUiMode("simple");
      if (!isSimpleRoute) {
        router.replace("/simple");
      }
      return;
    }

    if (requestedMode === "expert") {
      setUiMode("expert");
      if (isSimpleRoute) {
        router.replace(expertRouteForSimpleStep(activeSimpleStep));
      }
    }
  }, [activeSimpleStep, isSimpleRoute, router, setUiMode]);

  const toggleMode = () => {
    if (uiMode === "expert") {
      setUiMode("simple");
      router.push("/simple");
      return;
    }

    setUiMode("expert");
    if (isSimpleRoute) {
      router.push(expertRouteForSimpleStep(activeSimpleStep));
    }
  };

  return (
    <header className="studioNav appTopNav" role="navigation" aria-label="Primary">
      <Link
        className="studioNavBrand appTopNavBrandLink"
        href={uiMode === "simple" ? "/simple" : "/"}
      >
        <span className="studioNavDot" aria-hidden="true" />
        <span>LLM Builder</span>
      </Link>

      <nav className="studioNavLinks" aria-label="Primary routes">
        {links.map((link) => (
          <Link
            key={link.id}
            className="studioNavLink"
            href={link.href}
            aria-current={isRouteActive(pathname, link.href) ? "page" : undefined}
          >
            {link.label}
          </Link>
        ))}
      </nav>

      <div className="studioNavControls">
        <button
          type="button"
          className="modeSwitch"
          role="switch"
          aria-checked={uiMode === "simple"}
          aria-label={modeSwitchTitle}
          title={modeSwitchTitle}
          onClick={toggleMode}
        >
          <span className="modeSwitchTrack" aria-hidden="true">
            <span className="modeSwitchThumb" />
          </span>
          <span>{modeSwitchLabel}</span>
        </button>

        <button
          type="button"
          className="themeToggle"
          onClick={() => setTheme((current) => (current === "dark" ? "white" : "dark"))}
          aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
          title={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
        >
          {theme === "dark" ? <FiSun aria-hidden="true" /> : <FiMoon aria-hidden="true" />}
        </button>
      </div>
    </header>
  );
}
