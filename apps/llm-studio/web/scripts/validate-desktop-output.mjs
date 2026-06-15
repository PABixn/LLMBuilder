import { readFile, readdir, stat } from "node:fs/promises";
import path from "node:path";

const outputRoot = path.resolve("out");
const expectedRoutes = ["", "simple", "studio", "tokenizer", "training", "inference"];
const forbiddenPatterns = [
  /\/Users\/[^/]+\//,
  /\/home\/[^/]+\//,
  /[A-Z]:\\Users\\[^\\]+\\/i,
  /-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----/,
  /LLM_STUDIO_RUNTIME_TOKEN\s*=/,
  /(?:RUNPOD_)?API_KEY\s*=\s*["']?(?:rpa|rps)_[a-z0-9_-]{16,}/i,
  /Authorization\s*:\s*Bearer\s+[a-z0-9._~+/-]{24,}/i,
  /(?:localStorage|sessionStorage)\s*\.\s*setItem\s*\(\s*["'][^"']*(?:runtime[-_]?token|auth[-_]?token|access[-_]?token|refresh[-_]?token|pod[-_]?token|runpod[-_]?key|api[-_]?key|secret)[^"']*["']/i,
];
const configuredRuntimeToken = process.env.NEXT_PUBLIC_RUNTIME_TOKEN?.trim();
if (configuredRuntimeToken) {
  forbiddenPatterns.push(new RegExp(escapeRegExp(configuredRuntimeToken)));
}

const missing = [];
for (const route of expectedRoutes) {
  const routePath = route
    ? path.join(outputRoot, route, "index.html")
    : path.join(outputRoot, "index.html");
  try {
    if (!(await stat(routePath)).isFile()) {
      missing.push(routePath);
    }
  } catch {
    missing.push(routePath);
  }
}

if (missing.length) {
  throw new Error(`Desktop output is missing expected routes:\n${missing.join("\n")}`);
}

for (const file of await walk(outputRoot)) {
  if (!/\.(?:html|js|css|json|txt|map)$/i.test(file)) {
    continue;
  }
  const content = await readFile(file, "utf8");
  for (const pattern of forbiddenPatterns) {
    if (pattern.test(content)) {
      throw new Error(`Desktop output contains forbidden pattern ${pattern} in ${file}`);
    }
  }
}

console.log(`Validated desktop static output with ${expectedRoutes.length} routes.`);

async function walk(directory) {
  const files = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const resolved = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      files.push(...(await walk(resolved)));
    } else if (entry.isFile()) {
      files.push(resolved);
    }
  }
  return files;
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
