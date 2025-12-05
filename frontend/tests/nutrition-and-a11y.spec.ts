import { test, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

// Nutrition flow generates a plan and renders macros
test('nutrition plan generation works', async ({ page }) => {
  await page.goto('http://localhost:5173/login');
  // Attempt login if a test account exists; otherwise skip to nutrition
  try {
    await page.getByLabel('Email Address').fill('qa@example.com');
    await page.getByLabel('Password').fill('Password123!');
    await page.getByRole('button', { name: 'Sign In' }).click();
    await page.waitForURL('**/dashboard', { timeout: 5000 });
  } catch {}

  await page.goto('http://localhost:5173/nutrition');
  await page.getByRole('button', { name: 'Generate Plan' }).click();
  await expect(page.getByText(/Daily Targets/)).toBeVisible({ timeout: 15000 });
  await expect(page.getByText(/kcal/)).toBeVisible();
});

// Accessibility check on Dashboard
test('dashboard has no critical accessibility violations', async ({ page }) => {
  await page.goto('http://localhost:5173/dashboard');
  const accessibilityScanResults = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa'])
    .analyze();

  const critical = accessibilityScanResults.violations.filter(v => (v.impact || '').toLowerCase() === 'critical');
  expect(critical, JSON.stringify(critical, null, 2)).toEqual([]);
});
