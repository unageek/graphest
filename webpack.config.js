const ESLintPlugin = require("eslint-webpack-plugin");
const HtmlWebpackPlugin = require("html-webpack-plugin");
const path = require("path");
const process = require("process");

function baseConfig() {
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
      throw new Error("specify NODE_ENV={development|production}");
  }
}

const tsLoaderRule = {
  include: path.resolve(__dirname, "src"),
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
    entry: "./src/main.ts",
    module: {
      rules: [tsLoaderRule],
    },
    output: {
      path: path.join(__dirname, "dist"),
      filename: "main.js",
    },
    plugins,
  };
}

function preloadConfig() {
  return {
    ...baseConfig(),
    target: "electron-preload",
    entry: "./src/preload.ts",
    module: {
      rules: [tsLoaderRule],
    },
    output: {
      path: path.join(__dirname, "dist"),
      filename: "preload.js",
    },
    plugins,
  };
}

function rendererConfig() {
  return {
    ...baseConfig(),
    target: "electron-renderer",
    entry: {
      bundle: "./src/App.tsx",
    },
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
    output: {
      path: path.join(__dirname, "dist"),
      filename: "[name].js",
    },
    plugins: [
      ...plugins,
      new HtmlWebpackPlugin({
        template: "./src/index.html",
      }),
    ],
  };
}

module.exports = [mainConfig(), preloadConfig(), rendererConfig()];
