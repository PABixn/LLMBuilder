"use client";

import { useEffect, useState } from "react";

import {
  configNumberInputMode,
  configNumberInputPattern,
  formatNumberInputValue,
  parseConfigNumberInput,
  sanitizeConfigNumberInput,
  type ConfigNumberMode,
} from "../lib/configNumber";

type BaseConfigNumberInputProps = {
  mode?: ConfigNumberMode;
  step?: number | string;
  min?: number | string;
  max?: number | string;
  placeholder?: string;
};

type ConfigNumberInputProps = BaseConfigNumberInputProps & {
  value: number;
  onCommit: (value: number) => void;
};

type OptionalConfigNumberInputProps = BaseConfigNumberInputProps & {
  value: number | null;
  onCommit: (value: number | null) => void;
};

export function ConfigNumberInput({
  value,
  onCommit,
  mode = "integer",
  step,
  min,
  max,
  placeholder,
}: ConfigNumberInputProps) {
  const [draft, setDraft] = useState(() => formatNumberInputValue(value, mode));
  const [focused, setFocused] = useState(false);
  const formattedValue = formatNumberInputValue(value, mode);

  useEffect(() => {
    if (!focused) {
      setDraft(formattedValue);
    }
  }, [focused, formattedValue]);

  return (
    <input
      type="text"
      inputMode={configNumberInputMode(mode)}
      pattern={configNumberInputPattern(mode)}
      step={step}
      min={min}
      max={max}
      placeholder={placeholder}
      value={draft}
      onFocus={() => setFocused(true)}
      onChange={(event) => {
        setDraft(sanitizeConfigNumberInput(event.target.value, mode));
      }}
      onBlur={() => {
        setFocused(false);
        const parsed = parseConfigNumberInput(draft, mode);
        if (parsed === null) {
          setDraft(formattedValue);
          return;
        }
        onCommit(parsed);
        setDraft(formatNumberInputValue(parsed, mode));
      }}
    />
  );
}

export function OptionalConfigNumberInput({
  value,
  onCommit,
  mode = "integer",
  step,
  min,
  max,
  placeholder,
}: OptionalConfigNumberInputProps) {
  const [draft, setDraft] = useState(() =>
    value === null ? "" : formatNumberInputValue(value, mode)
  );
  const [focused, setFocused] = useState(false);
  const formattedValue = value === null ? "" : formatNumberInputValue(value, mode);

  useEffect(() => {
    if (!focused) {
      setDraft(formattedValue);
    }
  }, [focused, formattedValue]);

  return (
    <input
      type="text"
      inputMode={configNumberInputMode(mode)}
      pattern={configNumberInputPattern(mode)}
      step={step}
      min={min}
      max={max}
      placeholder={placeholder}
      value={draft}
      onFocus={() => setFocused(true)}
      onChange={(event) => {
        setDraft(sanitizeConfigNumberInput(event.target.value, mode));
      }}
      onBlur={() => {
        setFocused(false);
        if (draft.trim() === "") {
          onCommit(null);
          setDraft("");
          return;
        }
        const parsed = parseConfigNumberInput(draft, mode);
        if (parsed === null) {
          setDraft(formattedValue);
          return;
        }
        onCommit(parsed);
        setDraft(formatNumberInputValue(parsed, mode));
      }}
    />
  );
}
