# AI-GP Simulator Local Setup

Last verified: 2026-06-05.

## Installed Paths

- Source zip: `C:\Users\JefferyWhitmire\Downloads\AI-GP Simulator v1.0.3364.zip`
- Extracted kit: `C:\Users\JefferyWhitmire\Desktop\Shared\AI-GP-Simulator-v1.0.3364`
- Simulator exe: `C:\Users\JefferyWhitmire\Desktop\Shared\AI-GP-Simulator-v1.0.3364\Simulator\FlightSim.exe`
- Python example: `C:\Users\JefferyWhitmire\Desktop\Shared\AI-GP-Simulator-v1.0.3364\PyAIPilotExample`
- Python venv: `C:\Users\JefferyWhitmire\Desktop\Shared\AI-GP-Simulator-v1.0.3364\PyAIPilotExample\.venv`

## Setup Completed

- Extracted the outer zip.
- Extracted `AIGP_3364.zip` into `Simulator`.
- Extracted `PyAIPilotExample.zip` into `PyAIPilotExample`.
- Created a Python 3.14 virtual environment.
- Installed `requirements.txt`.
- Verified Python compile and dependency imports.
- Verified `FlightSim.exe` starts and launches `DCGame-Win64-Shipping.exe`.

## Run

Launch simulator:

```powershell
Start-Process -FilePath 'C:\Users\JefferyWhitmire\Desktop\Shared\AI-GP-Simulator-v1.0.3364\Simulator\FlightSim.exe' -WorkingDirectory 'C:\Users\JefferyWhitmire\Desktop\Shared\AI-GP-Simulator-v1.0.3364\Simulator'
```

Run the sample Python pilot after the simulator is open and logged in:

```powershell
cd 'C:\Users\JefferyWhitmire\Desktop\Shared\AI-GP-Simulator-v1.0.3364\PyAIPilotExample'
.\.venv\Scripts\python.exe main.py
```

The sample connects to `127.0.0.1:14550` and runs a continuous control loop. Stop it with `Ctrl+C`.
