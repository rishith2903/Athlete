import { test as base, expect } from '@playwright/test';
import AxeBuilder from '@axe-core/playwright';

// Extend base test with an a11y helper
export const test = base.extend({});
export { expect };

export async function expectNoCriticalA11yViolations(page) {
  const accessibilityScanResults = await new AxeBuilder({ page })
    .withTags(['wcag2a', 'wcag2aa'])
    .analyze();

  const critical = accessibilityScanResults.violations.filter(v =>
    ['critical', 'serious'].includes(v.impact || '')
  );

  expect(critical, `Accessibility violations: ${critical.map(v => `${v.id}: ${v.description}`).join('; ')}`).toHaveLength(0);
}
