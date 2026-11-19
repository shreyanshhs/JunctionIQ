# JunctionIQ

An interactive Python traffic-signal simulator for CCA problem #16.

## Run in VS Code

1. Unzip the project. In VS Code, choose **File > Open Folder** and select the **JunctionIQ** folder.
2. Open **Terminal > New Terminal** and run `python3 app.py` (Python 3.10+).
3. Wait for the training message to finish. Open **http://127.0.0.1:8000** in your browser.
4. To stop the server, press **Ctrl+C** in the terminal.

No packages, internet, or API keys are required. If you edit `index.html`, restart the server to load the edited file.

## Demo

- Select **East–west rush** and start playback. Pause, scrub the timeline, or step one second at a time.
- At a chosen timestamp, click **+ NS surge** to increase north–south traffic for up to 25 seconds. The same burst arrives at each controller. The timeline before the selected timestamp stays reproducible.
- Open the **AI decision log** and jump to a switch. Compare the two learned Q-values. The minimum green, maximum green, and all-red periods are enforced by the simulator.
- Run a **20-run benchmark** to compare average wait, peak lane queue, throughput, and lowest-wait runs on different random seeds. An exported run CSV includes all 180 seconds for each controller, even during playback.

## Source files

- `app.py` serves the page and handles simulation and benchmark requests.
- `traffic.py` models the junction, trains the Q-table, implements both baseline controllers, and calculates results.
- `index.html` renders the dashboard and handles playback, timeline controls, and charts.

Each simulated second contributes the number of vehicles still queued to the total wait. A lower total is better for this metric. The model covers a single junction with straight-through traffic and fixed discharge capacity. Results apply to this model.
