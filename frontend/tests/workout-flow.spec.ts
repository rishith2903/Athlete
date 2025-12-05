import { test, expect } from '@playwright/test';

// Assumes a logged-in state or a fresh session; adjust to your auth flow as needed

test('navigate workouts list → detail, start and complete', async ({ page }) => {
  // Try login if needed
  await page.goto('http://localhost:5173/login');
  try {
    await page.getByLabel('Email Address').fill('demo_user@example.com');
    await page.getByLabel('Password').fill('DemoPass123!');
    await page.getByRole('button', { name: 'Sign In' }).click();
    await page.waitForURL('**/dashboard', { timeout: 5000 });
  } catch {}

  // Go to workouts
  await page.goto('http://localhost:5173/workouts');
  // If no workouts exist, consider creating via UI or seed script
  const firstCard = page.locator('text=View Details').first();
  await firstCard.click();

  // Start workout if planned
  const startBtn = page.getByRole('button', { name: 'Start' });
  if (await startBtn.isVisible()) {
    await startBtn.click();
  }

  // Fill completion form and complete (when in progress)
  const completeBtn = page.getByRole('button', { name: 'Complete' });
  if (await completeBtn.isVisible()) {
    await page.getByLabel('Rating (1-5)').selectOption('5');
    await page.getByLabel('Perceived Exertion (1-10)').selectOption('7');
    await page.getByLabel('Notes').fill('QA automation run');
    await completeBtn.click();
  }

  // Verify status label
  await expect(page.locator('text=COMPLETED')).toBeVisible({ timeout: 15000 });

  // Back to list
  await page.getByRole('link', { name: 'All Workouts' }).click();
  await expect(page).toHaveURL(/.*\/workouts$/);
});
