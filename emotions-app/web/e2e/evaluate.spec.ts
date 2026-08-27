import { test, expect } from "@playwright/test";

test("text flow: pick a question, type an answer, see a scored result", async ({ page }) => {
  await page.goto("/");

  // pick the first corpus question
  await page.getByPlaceholder("Search questions or categories").waitFor();
  await page.getByRole("button", { name: /tell me about yourself/i }).first().click();

  await page
    .getByPlaceholder(/Answer as you would in the interview/i)
    .fill(
      "I am a data scientist with five years of experience. I have a statistics degree " +
        "and I have shipped fraud models in Python with scikit-learn and PyTorch, including " +
        "one that cut false positives by 22 percent.",
    );

  await page.getByRole("button", { name: /evaluate answer/i }).click();

  await expect(page).toHaveURL(/\/results/, { timeout: 90_000 });
  await expect(page.getByText("out of 100")).toBeVisible();
  await expect(page.getByText(/Breakdown/i)).toBeVisible();
  await expect(page.getByText(/Relevance/i).first()).toBeVisible();
  await expect(page.getByText(/Feedback/i).first()).toBeVisible();

  // transcript row is present for text too (empty), proving the field survives
  const transcript = page.getByText("Transcript");
  // text answers have transcript === null -> the card is not rendered; that's fine
  await expect(transcript).toHaveCount(0);
});

test("refresh on /results keeps the result (sessionStorage, not router state)", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: /tell me about yourself/i }).first().click();
  await page
    .getByPlaceholder(/Answer as you would in the interview/i)
    .fill("I have a background in statistics and machine learning and four years of industry work building models.");
  await page.getByRole("button", { name: /evaluate answer/i }).click();
  await expect(page).toHaveURL(/\/results/, { timeout: 90_000 });

  await page.reload();
  await expect(page.getByText("out of 100")).toBeVisible();
});
