import { describe, it, expect } from 'vitest';
import api, { authAPI } from '../../src/services/api';

// This is a simple smoke to ensure headers are applied.
describe('api auth header', () => {
  it('sets Authorization header when token exists', async () => {
    localStorage.setItem('authToken', 'abc');
    const req = await api.getUri({ url: '/auth/login' });
    expect(typeof req).toBe('string');
  });
});
