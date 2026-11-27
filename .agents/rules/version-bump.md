# Version bump rule

Whenever you make a user-visible change to the extension (UI, icons, behavior, fixes), bump the version before finishing.

- Use semantic versioning: patch (x.y.Z) for fixes and small tweaks, minor (x.Y.0) for new features, major (X.0.0) for breaking changes.
- Keep the version identical in all of these files:
  - `manifest.json` (source of truth) and `dist/manifest.json`
  - `package.json`
  - `package-lock.json` (top-level `version` and `packages[""].version`)
  - the version badge in `index.html` (`Manifest V3 · vX.Y.Z`)
- Current version: **1.1.0**. Update this line whenever the version changes.
