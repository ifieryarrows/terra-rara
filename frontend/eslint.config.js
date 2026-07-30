import js from "@eslint/js";
import tsParser from "@typescript-eslint/parser";
import tsPlugin from "@typescript-eslint/eslint-plugin";
import reactHooksPlugin from "eslint-plugin-react-hooks";
import reactRefreshPlugin from "eslint-plugin-react-refresh";
import globals from "globals";

export default [
  // Ignore generated outputs
  { ignores: ["dist/**", "node_modules/**"] },

  // Base JS recommended rules for all JS/TS files
  {
    files: ["**/*.{js,jsx,ts,tsx}"],
    ...js.configs.recommended,
  },

  // TypeScript + React rules
  {
    files: ["**/*.{ts,tsx}"],
    languageOptions: {
      parser: tsParser,
      parserOptions: {
        ecmaVersion: 2020,
        sourceType: "module",
        ecmaFeatures: { jsx: true },
      },
      globals: {
        ...globals.browser,
      },
    },
    plugins: {
      "@typescript-eslint": tsPlugin,
      "react-hooks": reactHooksPlugin,
      "react-refresh": reactRefreshPlugin,
    },
    rules: {
      // TypeScript recommended baseline
      ...tsPlugin.configs["eslint-recommended"].overrides?.[0]?.rules,
      ...tsPlugin.configs.recommended.rules,

      // React Hooks
      ...reactHooksPlugin.configs.recommended.rules,

      // React Refresh — off to allow non-component exports alongside components
      "react-refresh/only-export-components": "off",

      // ── Rule overrides to match previous project behaviour ──────────────
      // The project pre-dates strict TypeScript linting; these rules were
      // not enforced before and would require large-scale refactoring.
      // They can be tightened incrementally in future PRs.
      "@typescript-eslint/no-explicit-any": "off",
      "@typescript-eslint/no-empty-object-type": "off",

      // Allow intentionally-unused variables prefixed with _
      "@typescript-eslint/no-unused-vars": [
        "error",
        {
          vars: "all",
          args: "after-used",
          varsIgnorePattern: "^_",
          argsIgnorePattern: "^_",
          caughtErrorsIgnorePattern: "^_",
        },
      ],
    },
  },
];
