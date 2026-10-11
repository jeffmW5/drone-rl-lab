# Native Drone Command Center

Windows .NET Framework 4.8 / WinForms frontend. The Windows desktop shortcut
points to the built EXE under the current Codex outputs directory. Physics/training
use the existing Ubuntu-Laya environments and D:\drone-rl-lab checkout.

The app owns a private ephemeral dashboard connection and its own simulator port.
It displays live training metrics, paired evaluation and checkpoints. Flight images
come from a separately launched preview/playback, not the trainer's rollout batch.
V006 previews use the frozen task config. Selected completed policy runs can play.
Training repeats use V007 CUDA continuation; closing the app does not stop a trainer.

Build with the installed .NET SDK and source/DroneCommandCenter.csproj, or the
Windows Framework csc compiler with System.Windows.Forms, System.Drawing and
System.Web.Extensions references. Copy source/backend.py alongside the executable
inside source/; retain the icon at the executable's directory root.

Hardware flight commands are not implemented. Configuration paths currently target
this user's Windows/WSL installation. V006 physics remains uncalibrated.

