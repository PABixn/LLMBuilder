import Link from "next/link";
import type { Dispatch, SetStateAction } from "react";
import { FiMoon, FiSun } from "react-icons/fi";

import type { ThemeMode } from "../../types";

type StudioTopNavProps = {
  theme: ThemeMode;
  setTheme: Dispatch<SetStateAction<ThemeMode>>;
};

export function StudioTopNav({ theme, setTheme }: StudioTopNavProps) {
  return (
    <nav className="studioNav" aria-label="LLM Studio navigation">
      <div className="studioNavBrand">
        <span className="studioNavDot" />
        <span>LLM Builder</span>
      </div>
      <div className="studioNavLinks">
        <Link className="studioNavLink" href="/">
          Home
        </Link>
        <Link className="studioNavLink" href="/studio" aria-current="page">
          LLM Studio
        </Link>
        <Link className="studioNavLink" href="/tokenizer">
          Tokenizer Studio
        </Link>
      </div>
      <button
        type="button"
        className="themeToggle"
        onClick={() => setTheme((current) => (current === "dark" ? "white" : "dark"))}
        aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
        title={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
      >
        {theme === "dark" ? <FiSun /> : <FiMoon />}
      </button>
    </nav>
  );
}
