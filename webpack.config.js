import ESLintPlugin from "eslint-webpack-plugin";
import * as path from "path";
import * as process from "process";

/** @typedef { import('webpack').Configuration } WebpackConfig */

/** @type { () => WebpackConfig } */
function baseConfig() {
  return {
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
            "set either NODE_ENV=development or NODE_ENV=production"
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
}

/** @type { import('webpack').RuleSetRule } */
const tsLoaderRule = {
  include: path.resolve("src"),
  test: /\.ts$/,
  resolve: {
    extensions: [".ts", ".js"],
  },
  use: "ts-loader",
};

/** @type { import('webpack').WebpackPluginInstance[] } */
const plugins = [
  new ESLintPlugin({
    extensions: ["js", "ts", "tsx"],
    fix: true,
  }),
];

/** @type { () => WebpackConfig } */
function mainConfig() {
  return {
    ...baseConfig(),
    target: "electron-main",
    entry: "./src/main/main.ts",
    module: {
      rules: [tsLoaderRule],
    },
    plugins,
  };
}

/** @type { () => WebpackConfig } */
function preloadConfig() {
  return {
    ...baseConfig(),
    target: "electron-preload",
    entry: { preload: "./src/renderer/preload.ts" },
    module: {
      rules: [tsLoaderRule],
    },
    // https://www.electronjs.org/docs/latest/tutorial/esm
    experiments: {
      outputModule: false,
    },
    plugins,
  };
}

/** @type { () => WebpackConfig } */
function rendererConfig() {
  return {
    ...baseConfig(),
    target: "electron-renderer",
    entry: { bundle: "./src/renderer/App.tsx" },
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
        {
          ...tsLoaderRule,
          test: /\.tsx?$/,
          resolve: {
            extensions: [".ts", ".tsx", ".js"],
          },
          use: {
            loader: "ts-loader",
            options: {
              compilerOptions: {
                jsx:
                  process.env.NODE_ENV === "development"
                    ? "react-jsxdev"
                    : "react-jsx",
              },
            },
          },
        },
      ],
    },
    plugins,
  };
}

/** @type { WebpackConfig[] } */
export default [mainConfig(), preloadConfig(), rendererConfig()];
