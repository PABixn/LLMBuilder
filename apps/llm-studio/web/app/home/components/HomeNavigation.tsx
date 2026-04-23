import Link from "next/link";
import { FiMoon, FiSun } from "react-icons/fi";

import type { ThemeMode } from "../../../lib/theme";

type HomeNavigationProps = {
  theme: ThemeMode;
  onToggleTheme: () => void;
};

export function HomeNavigation({ theme, onToggleTheme }: HomeNavigationProps) {
  return (
    <nav className="studioNav" aria-label="Primary">
      <div className="studioNavBrand">
        <span className="studioNavDot" />
        <span>LLM Builder</span>
      </div>
      <div className="studioNavLinks">
        <Link className="studioNavLink" href="/" aria-current="page">
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
        <Link className="studioNavLink" href="/inference">
          Inference
        </Link>
      </div>
      <button className="themeToggle" onClick={onToggleTheme}>
        {theme === "dark" ? <FiSun /> : <FiMoon />}
      </button>
    </nav>
  );
}
