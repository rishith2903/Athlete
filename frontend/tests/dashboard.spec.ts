import { test, expect } from '@playwright/test';
import { expectNoCriticalA11yViolations } from './test-helpers';

// Dashboard data rendering tests
function mockDashboardAPIs(page) {
  page.route('**/api/progress**', async route => {
    const json = { ok: true };
    await route.fulfill({ json, status: 200, headers: { 'access-control-allow-origin': '*' } });
  });
  page.route('**/api/workouts**', async route => {
    const json = { items: [] };
    await route.fulfill({ json, status: 200, headers: { 'access-control-allow-origin': '*' } });
  });
  page.route('**/api/nutrition**', async route => {
    const json = { items: [] };
    await route.fulfill({ json, status: 200, headers: { 'access-control-allow-origin': '*' } });
  });
}

async function seedAuth(page) {
  await page.addInitScript(() => {
    const user = { name: 'QA User', email: 'qa@example.com' };
    localStorage.setItem('authToken', 'fake-token');
    localStorage.setItem('user', JSON.stringify(user));
  });
}

for (const device of ['desktop', 'mobile']) {
  test.describe(`Dashboard (${device})`, () => {
    test.use(device === 'mobile' ? { viewport: { width: 390, height: 844 } } : {});

    test.beforeEach(async ({ page }) => {
      mockDashboardAPIs(page);
      await seedAuth(page);
    });

    test('renders stats, charts, and recent workouts sections', async ({ page }) => {
      await page.goto('/dashboard');
      await expect(page.getByRole('heading', { name: /Welcome back/i })).toBeVisible();
      await expect(page.getByRole('heading', { name: 'Weight Progress' })).toBeVisible();
      await expect(page.getByRole('heading', { name: 'Calories Burned' })).toBeVisible();
      await expect(page.getByRole('heading', { name: 'Recent Workouts' })).toBeVisible();

      await expectNoCriticalA11yViolations(page);
    });

    test('quick actions navigate to pose analysis and chatbot', async ({ page }) => {
      await page.goto('/dashboard');
      await page.getByRole('link', { name: /Check Form/i }).click();
      await expect(page).toHaveURL(/pose-analysis/);
      await page.goBack();
      await page.getByRole('link', { name: /AI Coach/i }).click();
      await expect(page).toHaveURL(/chatbot/);
    });
  });
}
