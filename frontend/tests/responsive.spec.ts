import { test, expect } from '@playwright/test';

for (const device of ['desktop', 'mobile']) {
  test.describe(`Responsiveness (${device})`, () => {
    test.use(device === 'mobile' ? { viewport: { width: 390, height: 844 } } : {});

    test('landing hero and dashboard grids adapt', async ({ page }) => {
      await page.goto('/');
      // Hero buttons visible and stacked on mobile
      await expect(page.getByRole('link', { name: /Start Free Trial/i })).toBeVisible();

      // Auth and go to dashboard
      await page.addInitScript(() => {
        const user = { name: 'QA User', email: 'qa@example.com' };
        localStorage.setItem('authToken', 'fake-token');
        localStorage.setItem('user', JSON.stringify(user));
      });
      await page.goto('/dashboard');

      // Ensure key grid sections render without overflow (smoke check by visibility)
      await expect(page.getByRole('heading', { name: 'Weight Progress' })).toBeVisible();
      await expect(page.getByRole('heading', { name: 'Recent Workouts' })).toBeVisible();
    });
  });
}
