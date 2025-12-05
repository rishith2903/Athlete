import { test, expect } from '@playwright/test';
import { expectNoCriticalA11yViolations } from './test-helpers';

// Mock network helpers
function mockAuthEndpoints(page) {
  page.route('**/api/auth/login', async route => {
    const json = { token: 'fake-token', user: { name: 'QA User', email: 'qa@example.com' } };
    await route.fulfill({ json, status: 200, headers: { 'access-control-allow-origin': '*' } });
  });
  page.route('**/api/auth/signup', async route => {
    const json = { token: 'fake-token', user: { name: 'QA User', email: 'qa@example.com' } };
    await route.fulfill({ json, status: 200, headers: { 'access-control-allow-origin': '*' } });
  });
  page.route('**/api/auth/profile', async route => {
    const json = { name: 'QA User', email: 'qa@example.com' };
    await route.fulfill({ json, status: 200, headers: { 'access-control-allow-origin': '*' } });
  });
}

function mockOtherAPIs(page) {
  page.route('**/api/**', async route => {
    // Default pass-through for any API not explicitly mocked
    await route.continue();
  });
}

// Authentication tests
for (const device of ['desktop', 'mobile']) {
  test.describe(`Auth (${device})`, () => {
    test.use(device === 'mobile' ? { viewport: { width: 390, height: 844 } } : {});

    test.beforeEach(async ({ page }) => {
      mockOtherAPIs(page);
      mockAuthEndpoints(page);
    });

    test('Login validation and success flow', async ({ page }) => {
      await page.goto('/login');

      // Validation errors for empty submit
      await page.getByRole('button', { name: /sign in/i }).click();
      await expect(page.getByText('Email is required')).toBeVisible();
      await expect(page.getByText('Password is required')).toBeVisible();

      // Invalid email format
      await page.getByLabel('Email Address').fill('invalid');
      await page.getByLabel('Password').fill('password123');
      await page.getByRole('button', { name: /sign in/i }).click();
      await expect(page.getByText('Email is invalid')).toBeVisible();

      // Correct form
      await page.getByLabel('Email Address').fill('qa@example.com');
      await page.getByLabel('Password').fill('password123');
      await page.getByRole('button', { name: /sign in/i }).click();

      await expect(page).toHaveURL(/\/dashboard$/);
      await expectNoCriticalA11yViolations(page);
    });

    test('Signup validation and success flow', async ({ page }) => {
      await page.goto('/signup');

      // Empty submit
      await page.getByRole('button', { name: /sign up/i }).click();
      await expect(page.getByText('Name is required')).toBeVisible();
      await expect(page.getByText('Email is required')).toBeVisible();
      await expect(page.getByText('Password is required')).toBeVisible();
      await expect(page.getByText('Please confirm your password')).toBeVisible();

      // Fill with mismatched passwords
      await page.getByLabel('Full Name').fill('QA User');
      await page.getByLabel('Email Address').fill('qa@example.com');
      await page.getByLabel('Password').fill('password123');
      await page.getByLabel('Confirm Password').fill('password124');
      await page.getByRole('button', { name: /sign up/i }).click();
      await expect(page.getByText('Passwords do not match')).toBeVisible();

      // Correct passwords and required fields
      await page.getByLabel('Confirm Password').fill('password123');
      await page.getByLabel('Date of Birth').fill('1990-01-01');
      // Accept terms
      await page.getByLabel('I agree to the Terms and Conditions').check();

      await page.getByRole('button', { name: /sign up/i }).click();
      await expect(page).toHaveURL(/\/dashboard$/);
      await expectNoCriticalA11yViolations(page);
    });
  });
}
