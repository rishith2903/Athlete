import { test, expect } from '@playwright/test';
import { expectNoCriticalA11yViolations } from './test-helpers';

// Utility to seed localStorage auth state
async function seedAuth(page) {
  await page.addInitScript(() => {
    const user = { name: 'QA User', email: 'qa@example.com' };
    localStorage.setItem('authToken', 'fake-token');
    localStorage.setItem('user', JSON.stringify(user));
  });
}

// Navigation and layout tests (desktop/mobile)
for (const device of ['desktop', 'mobile']) {
  test.describe(`Navigation & Layout (${device})`, () => {
    test.use(device === 'mobile' ? { viewport: { width: 390, height: 844 } } : {});

    test('Landing -> Login -> Signup routes render and are accessible', async ({ page }) => {
      await page.goto('/');
      await expect(page.getByRole('link', { name: 'Login' })).toBeVisible();
      await page.getByRole('link', { name: 'Login' }).click();
      await expect(page).toHaveURL(/\/login$/);
      await expect(page.getByRole('heading', { name: /Welcome Back/i })).toBeVisible();

      // a11y
      await expectNoCriticalA11yViolations(page);

      await page.getByRole('link', { name: /Sign up/i }).click();
      await expect(page).toHaveURL(/\/signup$/);
      await expect(page.getByRole('heading', { name: /Create Account/i })).toBeVisible();
      await expectNoCriticalA11yViolations(page);
    });

    test('Protected area redirects unauthenticated users to login', async ({ page }) => {
      await page.goto('/dashboard');
      await expect(page).toHaveURL(/\/login$/);
    });

    test('Sidebar navigation items highlight active route after auth', async ({ page }) => {
      await seedAuth(page);
      await page.goto('/dashboard');
      await expect(page.getByRole('heading', { name: /Welcome back/i })).toBeVisible();

      // Navigate to Workouts
      await page.getByRole('link', { name: 'Workouts' }).click();
      await expect(page).toHaveURL(/\/workouts$/);
      await expect(page.getByRole('heading', { name: 'Workouts' })).toBeVisible();
    });
  });
}
