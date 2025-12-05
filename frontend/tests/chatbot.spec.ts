import { test, expect } from '@playwright/test';
import { expectNoCriticalA11yViolations } from './test-helpers';

async function seedAuth(page) {
  await page.addInitScript(() => {
    const user = { name: 'QA User', email: 'qa@example.com' };
    localStorage.setItem('authToken', 'fake-token');
    localStorage.setItem('user', JSON.stringify(user));
  });
}

for (const device of ['desktop', 'mobile']) {
  test.describe(`Chatbot (${device})`, () => {
    test.use(device === 'mobile' ? { viewport: { width: 390, height: 844 } } : {});

    test.beforeEach(async ({ page }) => {
      await seedAuth(page);
    });

    test('send message and see bot response', async ({ page }) => {
      await page.goto('/chatbot');
      await expect(page.getByText('AI Fitness Coach')).toBeVisible();

      await page.getByPlaceholder('Type your message...').fill('Create a workout plan for weight loss');
      // Press Enter to send (component handles Enter key)
      await page.getByPlaceholder('Type your message...').press('Enter');

      // Wait for a bot message related to weight loss to appear
      await expect(page.getByText(/weight loss/i)).toBeVisible({ timeout: 7000 });

      await expectNoCriticalA11yViolations(page);
    });
  });
}
