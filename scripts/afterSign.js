require("dotenv").config();

const { notarize } = require("electron-notarize");
const path = require("path");
const build = require("../electron-builder.json");

module.exports = async function (params) {
  if (params.electronPlatformName !== "darwin") return;

  const appBundleId = build.appId;
  const appPath = path.join(
    params.appOutDir,
    `${params.packager.appInfo.productFilename}.app`
  );

  console.log(`Notarizing ${appBundleId} found at ${appPath}`);

  await notarize({
    appBundleId,
    appPath,
    appleId: process.env.notarizeAppleId,
    appleIdPassword: process.env.notarizeAppleIdPassword,
  });
};
