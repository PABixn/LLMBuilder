import Link from "next/link";
import { FiMoon, FiSun } from "react-icons/fi";

import type { ThemeMode } from "../../../lib/theme";

type TrainingStudioNavProps = {
  theme: ThemeMode;
  onToggleTheme: () => void;
};

export function TrainingStudioNav({
  theme,
  onToggleTheme,
}: TrainingStudioNavProps) {
  return (
    <header className="studioNav" role="navigation" aria-label="Primary">
      <div className="studioNavBrand">
        <span className="studioNavDot" />
        <span>LLM Builder</span>
      </div>
      <div className="studioNavLinks">
        <Link className="studioNavLink" href="/">
          Home
        </Link>
        <Link className="studioNavLink" href="/studio">
          LLM Studio
        </Link>
        <Link className="studioNavLink" href="/tokenizer">
          Tokenizer Studio
        </Link>
        <Link className="studioNavLink" href="/training" aria-current="page">
          Training
        </Link>
        <Link className="studioNavLink" href="/inference">
          Inference
        </Link>
      </div>
      <button
        type="button"
        className="themeToggle"
        onClick={onToggleTheme}
        aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
        title={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
      >
        {theme === "dark" ? <FiSun /> : <FiMoon />}
      </button>
    </header>
  );
}
