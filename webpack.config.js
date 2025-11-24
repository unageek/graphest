import ESLintPlugin from "eslint-webpack-plugin";
import HtmlWebpackPlugin from "html-webpack-plugin";
import * as path from "path";
import * as process from "process";

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
    output: {
      path: path.resolve("dist"),
      filename: "[name].js",
    },
  };
}

const tsLoaderRule = {
  include: path.resolve("src"),
  test: /\.ts$/,
  resolve: {
    extensions: [".ts", ".js"],
  },
  use: "ts-loader",
};

const plugins = [
  new ESLintPlugin({
    extensions: ["js", "ts", "tsx"],
    fix: true,
  }),
];

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

function preloadConfig() {
  return {
    ...baseConfig(),
    target: "electron-preload",
    entry: { preload: "./src/renderer/preload.ts" },
    module: {
      rules: [tsLoaderRule],
    },
    plugins,
  };
}

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
        },
      ],
    },
    plugins: [
      ...plugins,
      new HtmlWebpackPlugin({
        template: "src/renderer/index.html",
      }),
    ],
  };
}

export default [mainConfig(), preloadConfig(), rendererConfig()];
