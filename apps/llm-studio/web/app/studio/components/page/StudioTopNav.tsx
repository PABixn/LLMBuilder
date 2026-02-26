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
        <span>LLM Studio</span>
      </div>
      <div className="studioNavLinks">
        <a className="studioNavLink" href="#base-model">
          Base Model
        </a>
        <a className="studioNavLink" href="#block-builder">
          Builder
        </a>
        <a className="studioNavLink" href="#diagnostics">
          Diagnostics
        </a>
        <a className="studioNavLink" href="#model-analysis">
          Analysis
        </a>
        <a className="studioNavLink" href="#json-preview">
          JSON
        </a>
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
