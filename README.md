# Welcome to Brain
A high-fidelity spiking neuron simulator with a live visualization dashboard.

## Contributing
The project has been setup with uv and pre-commit.

### Setup
To setup with uv, simply execute:

`uv sync --dev`

This will install all the required dependencies for the project.

### Running
Run a short simulation in the terminal:

`uv run brain --steps 500`

Run the local visualization server (interactive dashboard with scenarios and circuit builder):

`uv run brain --server --auto-scale`

### Dashboard Highlights
- Neural field, topology sketch, raster plot, and E/I firing rate split.
- Scenario presets: Baseline, Burst/Quench, Oscillator, Ripple.
- Circuit builder: place pacemaker, bursting, and fast inhibitory neurons on the grid.
- Controls: pause/resume, step, pulse, noise/pulse sliders, and map mode toggle.

### Testing
To test with pre-commit, simply run the following to test the code:

`uv run pre-commit run --all-files`

## Learning
This is an old project I'm refactoring and using to learn new tools such as:
- mypy
- precommit
- pytest
