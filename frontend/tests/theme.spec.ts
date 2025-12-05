import { test, expect } from '@playwright/test';

for (const device of ['desktop', 'mobile']) {
  test.describe(`Theme toggle (${device})`, () => {
    test.use(device === 'mobile' ? { viewport: { width: 390, height: 844 } } : {});

    test('placeholder theme test - verify page loads without theme toggle', async ({ page }) => {
      await page.goto('/');
      await expect(page).toHaveTitle(/Vite \+ React/i);
    });
  });
}
