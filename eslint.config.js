import { includeIgnoreFile } from "@eslint/compat";
import js from "@eslint/js";
import ts from "@typescript-eslint/eslint-plugin";
import tsParser from "@typescript-eslint/parser";
import reactHooks from "eslint-plugin-react-hooks";
import globals from "globals";
import * as path from "node:path";

/** @type { import("eslint").Linter.Config[] } */
export default [
  js.configs.recommended,
  {
    files: ["*.js", "build/hooks/*.js", "src/**/*.ts", "src/**/*.tsx"],
    languageOptions: {
      parser: tsParser,
      parserOptions: {
        ecmaVersion: 2024,
      },
      globals: {
        ...globals.browser,
        ...globals.es2024,
        ...globals.jest,
        ...globals.node,
        Electron: "readonly",
        L: "readonly",
      },
    },
    plugins: {
      "@typescript-eslint": /** @type { any } */ (ts),
      "react-hooks": /** @type { any } */ (reactHooks),
    },
    rules: {
      ...ts.configs.recommended.rules,
      "no-unused-vars": "off",
      "@typescript-eslint/no-unused-vars": [
        "error",
        {
          args: "all",
          argsIgnorePattern: "^_",
          caughtErrors: "all",
          caughtErrorsIgnorePattern: "^_",
          destructuredArrayIgnorePattern: "^_",
          ignoreRestSiblings: true,
          vars: "all",
          varsIgnorePattern: "^_",
        },
      ],
      "@typescript-eslint/no-non-null-assertion": "off",
      eqeqeq: "error",
      "react-hooks/exhaustive-deps": "error",
      "react-hooks/rules-of-hooks": "error",
    },
  },
  includeIgnoreFile(path.resolve(".gitignore")),
];
