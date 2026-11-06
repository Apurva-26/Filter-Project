# glasses-filter-project-template
Team: Saumyaa, Tristan, Apurva, Timmy

## How this repo works
- **Issues** hold our user stories. Use the "User story" template.
- **constraints.md** holds the facts every story must respect.
- **test-cases.md** holds tests that trace back to stories by issue number.
- **prototype/** holds snapshots of our code when it passes a story's tests.

## Labels to create
- user: on purpose, user: passerby, user: maintainer, user: excluded
- priority: high, priority: medium, priority: low

## Running the filters
- `py launcher.py` shows the menu (tap a tile to start a filter, Back returns to the menu).
- On the Pi: `python launcher.py --fullscreen` (fullscreen + auto-back after 20 s with nobody there).
- The tiles are listed in `filters.json`. OpenCV filters use `kiosk.py` for the Back button and touch buttons.
