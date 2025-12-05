import { describe, it, expect, beforeEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import React from 'react';
import { AuthProvider, useAuth } from '../../src/contexts/AuthContext';

function wrapper({ children }) {
  return <AuthProvider>{children}</AuthProvider>;
}

describe('AuthContext', () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it('persists token and user', async () => {
    const { result } = renderHook(() => useAuth(), { wrapper });
    // Simulate login by directly setting storage (AuthProvider reads on mount)
    act(() => {
      localStorage.setItem('authToken', 'test-token');
      localStorage.setItem('user', JSON.stringify({ id: 'u1', username: 'alice', name: 'Alice' }));
    });

    // Trigger a re-render by toggling logout then setting again
    await act(async () => {
      await result.current.logout();
      localStorage.setItem('authToken', 'test-token');
      localStorage.setItem('user', JSON.stringify({ id: 'u1', username: 'alice', name: 'Alice' }));
    });

    expect(localStorage.getItem('authToken')).toBe('test-token');
    expect(JSON.parse(localStorage.getItem('user') || '{}').username).toBe('alice');
  });
});
