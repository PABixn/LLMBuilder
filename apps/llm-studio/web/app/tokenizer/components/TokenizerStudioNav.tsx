import Link from "next/link";
import { FiMoon, FiSun } from "react-icons/fi";

import type { ThemeMode } from "../../../lib/theme";

type TokenizerStudioNavProps = {
  themeMode: ThemeMode;
  onToggleTheme: () => void;
};

export function TokenizerStudioNav({
  themeMode,
  onToggleTheme,
}: TokenizerStudioNavProps) {
  return (
    <header className="studioNav" role="navigation" aria-label="Primary">
      <div className="studioNavBrand">
        <span className="studioNavDot" aria-hidden="true" />
        <span>LLM Builder</span>
      </div>
      <nav className="studioNavLinks" aria-label="Primary routes">
        <Link className="studioNavLink" href="/">
          Home
        </Link>
        <Link className="studioNavLink" href="/studio">
          LLM Studio
        </Link>
        <Link className="studioNavLink" href="/tokenizer" aria-current="page">
          Tokenizer Studio
        </Link>
        <Link className="studioNavLink" href="/training">
          Training
        </Link>
        <Link className="studioNavLink" href="/inference">
          Inference
        </Link>
      </nav>
      <button
        type="button"
        className="themeToggle"
        onClick={onToggleTheme}
        aria-label={
          themeMode === "dark"
            ? "Switch to white theme"
            : "Switch to dark theme"
        }
        title={
          themeMode === "dark"
            ? "Switch to white theme"
            : "Switch to dark theme"
        }
      >
        {themeMode === "dark" ? (
          <FiSun aria-hidden="true" />
        ) : (
          <FiMoon aria-hidden="true" />
        )}
      </button>
    </header>
  );
}
