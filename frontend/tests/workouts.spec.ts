import { test, expect } from '@playwright/test';

async function seedAuth(page) {
  await page.addInitScript(() => {
    const user = { name: 'QA User', email: 'qa@example.com' };
    localStorage.setItem('authToken', 'fake-token');
    localStorage.setItem('user', JSON.stringify(user));
  });
}

for (const device of ['desktop', 'mobile']) {
  test.describe(`Workouts (${device})`, () => {
    test.use(device === 'mobile' ? { viewport: { width: 390, height: 844 } } : {});

    test.beforeEach(async ({ page }) => {
      await seedAuth(page);
    });

    test('filters and search update workouts list', async ({ page }) => {
      await page.goto('/workouts');
      await expect(page.getByRole('heading', { name: 'Workouts' })).toBeVisible();

      // Search by name
      await page.getByPlaceholder('Search workouts...').fill('Yoga');
      await expect(page.getByText('Morning Yoga Flow')).toBeVisible();

      // Change category filter
      await page.getByRole('button', { name: 'Yoga' }).click();
      await expect(page.getByText('Morning Yoga Flow')).toBeVisible();
    });
  });
}
