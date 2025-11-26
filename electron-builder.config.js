/** @type { import("electron-builder").Configuration } */
export default {
  appId: "io.github.unageek.graphest",
  directories: {
    output: "release",
  },
  files: ["dist/**/*", "!**/*.js.map"],
  afterSign: "scripts/afterSign.js",
  protocols: [{ name: "Graphest", schemes: ["graphest"] }],
  mac: {
    category: "public.app-category.utilities",
    electronLanguages: ["en"],
    extendInfo: {
      CFBundleDocumentTypes: [
        {
          CFBundleTypeIconSystemGenerated: 1,
          CFBundleTypeRole: "Editor",
          LSHandlerRank: "Owner",
          LSItemContentTypes: ["io.github.unageek.graphest.document"],
        },
      ],
      UTExportedTypeDeclarations: [
        {
          UTTypeDescription: "Graphest Document",
          UTTypeIdentifier: "io.github.unageek.graphest.document",
          UTTypeConformsTo: ["public.content", "public.data"],
          UTTypeIcons: {
            UTTypeIconText: "graph",
          },
          UTTypeTagSpecification: {
            "public.filename-extension": "graphest",
          },
        },
      ],
    },
  },
  nsis: {
    oneClick: false,
    allowToChangeInstallationDirectory: true,
  },
  win: {
    fileAssociations: [{ ext: "graphest", name: "Graphest Document" }],
  },
};
