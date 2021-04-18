const path = require("path");
const ESLintPlugin = require("eslint-webpack-plugin");
const HtmlWebpackPlugin = require("html-webpack-plugin");

function baseConfig(mode) {
  switch (mode) {
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
      throw "mode must be either 'development' or 'production'";
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

const esLintPlugin = new ESLintPlugin({
  extensions: ["js", "ts", "tsx"],
  fix: true,
});

function mainConfig(mode) {
  return {
    ...baseConfig(mode),
    target: "electron-main",
    entry: "./src/main.ts",
    module: {
      rules: [tsLoaderRule],
    },
    output: {
      path: path.join(__dirname, "dist"),
      filename: "main.js",
    },
    plugins: [esLintPlugin],
  };
}

function preloadConfig(mode) {
  return {
    ...baseConfig(mode),
    target: "electron-preload",
    entry: "./src/preload.ts",
    module: {
      rules: [tsLoaderRule],
    },
    output: {
      path: path.join(__dirname, "dist"),
      filename: "preload.js",
    },
    plugins: [esLintPlugin],
  };
}

function rendererConfig(mode) {
  return {
    ...baseConfig(mode),
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
      esLintPlugin,
      new HtmlWebpackPlugin({
        template: "./src/index.html",
      }),
    ],
  };
}

module.exports = (env, argv) => {
  return [mainConfig, preloadConfig, rendererConfig].map((cfg) =>
    cfg(argv.mode)
  );
};
