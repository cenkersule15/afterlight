# Studio 02

The interface update preserves model training code, data, checkpoints and metrics. New controls:

- Persistent dark/light theme and pinned scenes (browser-local).
- Searchable library over the full bundled catalogue; partition and pinned-only filters.
- Wipe and side-by-side comparison, independent reference/prediction selection, opacity control.
- Focus mode hides nonessential header metrics and opens more space for the image workbench.
- Region class and minimum-area filters act on the region list, not the prediction mask; filtering does not change the model output.
- Region selection highlights the existing model bounding box. Raw model scores are uncalibrated.
- Scene-specific metrics are clearly distinct from aggregate held-out test metrics.
- Keyboard navigation and three language options.

Shortcuts: Cmd/Ctrl+K opens the library; left/right navigates scenes; R runs the selected model; L toggles reference labels; F toggles focus; Escape exits focus when no dialog is open. Form-field interactions are excluded.

## Validation

- TypeScript and production Vite build passed.
- Browser checks: real 66-scene catalogue, identifier search, scene pinning, pinned-only filter, scene selection, side-by-side/wipe mode, focus mode, dark/light theme, TR/EN/DE strings, U-Net inference, per-scene metrics, region class filtering and bounding-box selection.
- 214 bundled backend, data and artifact files match the previous release byte for byte.
- Docker is not installed in the authoring environment; Compose execution was not rerun here.

## Warm ivory appearance

Default workspace changed to warm ivory, with off-white cards and petrol-teal accents and sidebar. Dark mode remains available. A new browser preference key selects the updated appearance on first launch without affecting pinned scenes or language. Model/data assets and inference are unchanged.
