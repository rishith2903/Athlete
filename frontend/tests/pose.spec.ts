import { test, expect } from '@playwright/test';
import { expectNoCriticalA11yViolations } from './test-helpers';

function mockPoseAPIs(page) {
  page.route('**/api/pose/**', async route => {
    await route.fulfill({ json: { ok: true }, status: 200, headers: { 'access-control-allow-origin': '*' } });
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
  test.describe(`Pose Analysis (${device})`, () => {
    test.use(device === 'mobile' ? { viewport: { width: 390, height: 844 } } : {});

    test.beforeEach(async ({ page, context }) => {
      await context.grantPermissions(['camera']);
      // Mock getUserMedia to avoid using a real camera
      await page.addInitScript(() => {
        const MockMediaStream = function () {
          return {
            getTracks: () => [{ stop: () => {} }],
          };
        };
        navigator.mediaDevices = navigator.mediaDevices || ({});
        navigator.mediaDevices.getUserMedia = async () => new MockMediaStream();
      });
      mockPoseAPIs(page);
      await seedAuth(page);
    });

    test('start/stop camera and start/stop analysis, feedback overlay appears', async ({ page }) => {
      await page.goto('/pose-analysis');
      // Start camera
      await page.getByRole('button', { name: /Start Camera/i }).click();

      // Start Analysis -> expect button toggles to Stop Analysis and feedback appears eventually
      await page.getByRole('button', { name: /Start Analysis/i }).click();
      await expect(page.getByRole('button', { name: /Stop Analysis/i })).toBeVisible();

      // Feedback overlay (text changes randomly; just assert container appears eventually)
      await expect(page.getByText(/Reps:/)).toBeVisible();

      // Stop analysis
      await page.getByRole('button', { name: /Stop Analysis/i }).click();
      await page.getByRole('button', { name: /Stop Camera/i }).click();

      await expectNoCriticalA11yViolations(page);
    });
  });
}
