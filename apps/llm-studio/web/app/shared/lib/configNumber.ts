export type ConfigNumberMode = "integer" | "decimal" | "scientific";

export function sanitizePositiveIntegerInput(value: string): string {
  return value.replace(/[^0-9]/g, "");
}

export function sanitizePositiveDecimalInput(value: string): string {
  const digitsAndDot = value.replace(/[^0-9.]/g, "");
  const firstDotIndex = digitsAndDot.indexOf(".");
  if (firstDotIndex === -1) {
    return digitsAndDot;
  }
  return `${digitsAndDot.slice(0, firstDotIndex + 1)}${digitsAndDot
    .slice(firstDotIndex + 1)
    .replace(/\./g, "")}`;
}

export function sanitizePositiveScientificInput(value: string): string {
  const compact = value.replace(/,/g, "").replace(/\s/g, "").toLowerCase();
  const exponentIndex = compact.indexOf("e");
  const mantissaRaw = exponentIndex === -1 ? compact : compact.slice(0, exponentIndex);
  const exponentRaw = exponentIndex === -1 ? "" : compact.slice(exponentIndex + 1);

  let mantissa = "";
  let hasDot = false;
  for (const char of mantissaRaw) {
    if (/[0-9]/.test(char)) {
      mantissa += char;
      continue;
    }
    if (char === "." && !hasDot) {
      mantissa += char;
      hasDot = true;
    }
  }

  if (exponentIndex === -1) {
    return mantissa;
  }

  let exponent = "";
  let hasSign = false;
  for (const char of exponentRaw) {
    if (/[0-9]/.test(char)) {
      exponent += char;
      continue;
    }
    if ((char === "-" || char === "+") && !hasSign && exponent === "") {
      exponent += char;
      hasSign = true;
    }
  }

  return `${mantissa}e${exponent}`;
}

export function sanitizeConfigNumberInput(value: string, mode: ConfigNumberMode): string {
  if (mode === "scientific") {
    return sanitizePositiveScientificInput(value);
  }
  if (mode === "decimal") {
    return sanitizePositiveDecimalInput(value);
  }
  return sanitizePositiveIntegerInput(value);
}

export function formatExponentialValue(value: number, digits = 3): string {
  if (!Number.isFinite(value)) {
    return "";
  }
  const [mantissa = "", exponent = "0"] = value.toExponential(digits).split("e");
  const trimmedMantissa = mantissa.replace(/\.?0+$/, "");
  const exponentValue = Number(exponent);
  const exponentSign = exponentValue >= 0 ? "+" : "";
  return `${trimmedMantissa}e${exponentSign}${exponentValue}`;
}

export function formatNumberInputValue(
  value: number,
  mode: ConfigNumberMode = "integer"
): string {
  if (!Number.isFinite(value)) {
    return "";
  }
  if (mode === "scientific") {
    return formatExponentialValue(value, 3);
  }
  const asText = String(value);
  if (!/[eE]/.test(asText)) {
    return asText;
  }
  return value.toLocaleString("en-US", {
    useGrouping: false,
    maximumFractionDigits: 20,
  });
}

export function parseConfigNumberInput(
  value: string,
  mode: ConfigNumberMode
): number | null {
  const trimmed = value.trim();
  if (trimmed === "" || trimmed === "." || /e[+-]?$/i.test(trimmed)) {
    return null;
  }
  const parsed = Number(trimmed);
  if (!Number.isFinite(parsed)) {
    return null;
  }
  if (mode === "integer" && !Number.isInteger(parsed)) {
    return null;
  }
  return parsed;
}

export function configNumberInputPattern(mode: ConfigNumberMode): string {
  if (mode === "scientific") {
    return "[0-9]*[.]?[0-9]*([eE][+-]?[0-9]+)?";
  }
  if (mode === "decimal") {
    return "[0-9]*[.]?[0-9]*";
  }
  return "[0-9]*";
}

export function configNumberInputMode(mode: ConfigNumberMode): "numeric" | "decimal" {
  return mode === "integer" ? "numeric" : "decimal";
}
