# nzmedic.github.io
Decision cockpit concepts for decision-focused analytics and applied data science

## Repo structure
- `assets/`: shared CSS, images, and fonts for all cockpits.
- `cockpits/`: static cockpit front-ends pulling data from static output files
  - `shared/`: shared JS/HTML shell utilities for cockpit pages.
  - `<cockpit-name>/`: cockpit-specific HTML/JS plus an `outputs/` folder.
- `projects/`: model/data pipelines that generate cockpit outputs.
  - `<project-name>/`: project-specific notebooks/scripts/source.
  - `shared/`: shared feature helpers or utilities reused across projects.
- `js/`: global JS utilities used by multiple pages (formatting, model helpers).

## Shared resources
- Shared cockpit navigation and utilities live in `cockpits/shared/` to avoid HTML and JS duplication.
- Styling stays centralized in `assets/css/cockpit.css` so each cockpit can reference the same UI primitives.
- Python feature helpers live in `projects/lendy-graduation-retention/feature_utils.py` and can be extended into `projects/shared/` if additional cross-project utilities emerge.
