import Link from "next/link";
import { FiMoon, FiSun } from "react-icons/fi";

import type { ThemeMode } from "../../../lib/theme";

type InferenceTopNavProps = {
  theme: ThemeMode;
  onToggleTheme: () => void;
};

export function InferenceTopNav({ theme, onToggleTheme }: InferenceTopNavProps) {
  return (
    <header className="studioNav" role="navigation" aria-label="Primary">
      <div className="studioNavBrand">
        <span className="studioNavDot" />
        <span>LLM Builder</span>
      </div>
      <nav className="studioNavLinks" aria-label="Primary routes">
        <Link className="studioNavLink" href="/">
          Home
        </Link>
        <Link className="studioNavLink" href="/studio">
          LLM Studio
        </Link>
        <Link className="studioNavLink" href="/tokenizer">
          Tokenizer Studio
        </Link>
        <Link className="studioNavLink" href="/training">
          Training
        </Link>
        <Link className="studioNavLink" href="/inference" aria-current="page">
          Inference
        </Link>
      </nav>
      <button className="themeToggle" onClick={onToggleTheme} aria-label="Toggle theme">
        {theme === "dark" ? <FiSun /> : <FiMoon />}
      </button>
    </header>
  );
}
