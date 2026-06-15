import path from "node:path";

import { expect, test, type Page } from "@playwright/test";

const repositoryRoot = path.resolve(__dirname, "../../../../..");
const screenshotRoot = path.join(repositoryRoot, "build/desktop/playwright/screenshots");
const routes = [
  { path: "/", name: "workspace" },
  { path: "/simple", name: "simple" },
  { path: "/studio", name: "studio" },
  { path: "/tokenizer", name: "tokenizer" },
  { path: "/training", name: "training" },
  { path: "/inference", name: "inference" },
] as const;

for (const route of routes) {
  test(`${route.path} loads directly without page or console errors`, async ({ page }) => {
    const errors = captureErrors(page);

    await page.goto(route.path);
    await expect(page.locator("body")).not.toContainText("Starting LLM Studio...");
    await expect(page.locator("body")).not.toContainText("LLM Studio could not start.");
    await expect(page.locator("main")).toBeVisible();
    await expect(page.locator("body")).not.toContainText("Internal Server Error");
    await expectNoHorizontalDocumentOverflow(page);
    await page.screenshot({
      path: path.join(screenshotRoot, `${route.name}.png`),
      fullPage: true,
    });

    expect(errors).toEqual([]);
  });
}

test("desktop query deep links remain on their intended routes", async ({ page }) => {
  for (const deepLink of [
    "/studio?project=missing-project",
    "/tokenizer?job=missing-job",
    "/training?project=missing-project&tokenizerJob=missing-job&run=missing-run",
  ]) {
    await page.goto(deepLink);
    await expect(page).toHaveURL(new RegExp(deepLink.split("?")[0].replace("/", "\\/")));
    await expect(page.locator("main")).toBeVisible();
  }
});

test("bundled-origin browser state survives navigation and reload", async ({ page }) => {
  await page.goto("/");
  await page.evaluate(() => localStorage.setItem("llm-studio-e2e-persistence", "retained"));
  await page.goto("/simple");
  await page.reload();

  await expect
    .poll(() => page.evaluate(() => localStorage.getItem("llm-studio-e2e-persistence")))
    .toBe("retained");
});

test("minimum desktop viewport renders every route without document overflow", async ({ page }) => {
  await page.setViewportSize({ width: 1080, height: 720 });
  for (const route of routes) {
    await page.goto(route.path);
    await expectNoHorizontalDocumentOverflow(page);
  }
});

function captureErrors(page: Page): string[] {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(`pageerror: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error") {
      errors.push(`console: ${message.text()}`);
    }
  });
  return errors;
}

async function expectNoHorizontalDocumentOverflow(page: Page): Promise<void> {
  await expect
    .poll(() =>
      page.evaluate(
        () => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1
      )
    )
    .toBe(true);
}
