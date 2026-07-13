import { defineConfig, configDefaults } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['test/**/*.test.ts'],
    exclude: [
      ...configDefaults.exclude,
      ...(process.env.LIVE_TESTS ? [] : ['test/live/**']),
    ],
    testTimeout: 30_000,
  },
});
