import * as path from "node:path";
import * as process from "node:process";

/** @typedef { import('webpack').Configuration } WebpackConfig */

/** @type { import('webpack').RuleSetRule } */
const tsLoaderRule = {
  include: path.resolve("src"),
  test: /\.tsx?$/,
  resolve: {
    extensions: [".js", ".jsx", ".ts", ".tsx"],
  },
  use: {
    loader: "ts-loader",
    options: {
      compilerOptions: {
        jsx:
          process.env.NODE_ENV === "development" ? "react-jsxdev" : "react-jsx",
      },
    },
  },
};

/** @type { WebpackConfig } */
const baseConfig = {
  ...(() => {
    switch (process.env.NODE_ENV) {
      case "development":
        return {
          devtool: "cheap-source-map",
          mode: "development",
          stats: "errors-warnings",
        };
      case "production":
        return {
          mode: "production",
          stats: "errors-warnings",
        };
      default:
        throw new Error(
          "set either NODE_ENV=development or NODE_ENV=production",
        );
    }
  })(),
  experiments: {
    outputModule: true,
  },
  output: {
    path: path.resolve("dist"),
    filename: "[name].js",
  },
};

/** @type { WebpackConfig } */
const mainConfig = {
  ...baseConfig,
  target: "electron-main",
  entry: path.resolve("src/main/main.ts"),
  module: {
    rules: [tsLoaderRule],
  },
};

/** @type { WebpackConfig } */
const preloadConfig = {
  ...baseConfig,
  target: "electron-preload",
  entry: { preload: path.resolve("src/renderer/preload.ts") },
  module: {
    rules: [tsLoaderRule],
  },
  // https://www.electronjs.org/docs/latest/tutorial/esm
  experiments: {
    outputModule: false,
  },
};

/** @type { WebpackConfig } */
const rendererConfig = {
  ...baseConfig,
  target: "electron-renderer",
  entry: { bundle: path.resolve("src/renderer/App.tsx") },
  module: {
    rules: [
      {
        test: /\.css$/,
        use: ["style-loader", "css-loader"],
      },
      {
        test: /\.(eot|png|svg|ttf|woff|woff2)$/,
        type: "asset/resource",
      },
      tsLoaderRule,
    ],
  },
};

/** @type { WebpackConfig[] } */
export default [mainConfig, preloadConfig, rendererConfig];
