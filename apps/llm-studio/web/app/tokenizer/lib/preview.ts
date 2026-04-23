import type { TokenizerPreviewToken } from "../../../lib/tokenizerLegacyApi";
import type { PreviewSegment } from "../types";

export function hydratePreviewText(value: unknown, fallback: string): string {
  if (typeof value !== "string") {
    return fallback;
  }
  return value.slice(0, 50_000);
}

export function prettyJson(value: unknown): string {
  return JSON.stringify(value, null, 2);
}

export function makePreviewSegments(
  text: string,
  tokens: TokenizerPreviewToken[]
): PreviewSegment[] {
  if (text.length === 0) {
    return [];
  }

  const segments: PreviewSegment[] = [];
  let cursor = 0;

  for (const token of tokens) {
    const start = Math.max(0, Math.min(text.length, Math.trunc(token.start)));
    const end = Math.max(0, Math.min(text.length, Math.trunc(token.end)));

    if (end <= start || start < cursor) {
      continue;
    }

    if (start > cursor) {
      segments.push({
        kind: "plain",
        text: text.slice(cursor, start),
      });
    }

    segments.push({
      kind: "token",
      text: text.slice(start, end),
      token,
    });
    cursor = end;
  }

  if (cursor < text.length) {
    segments.push({
      kind: "plain",
      text: text.slice(cursor),
    });
  }

  return segments;
}

export function displayTokenLabel(value: string): string {
  return value
    .replaceAll("Ġ", "[space]")
    .replaceAll("▁", "[space]")
    .replaceAll("Ċ", "[\\n]")
    .replaceAll("\r\n", "[\\r\\n]\n")
    .replaceAll(" ", "[space]")
    .replaceAll("\n", "[\\n]\n")
    .replaceAll("\t", "[\\t]");
}

export function tokenHue(index: number): number {
  return (index * 37) % 360;
}
