# CUDA continuation -- 2026-10-11

User requested CUDA training. V006 was deliberately interrupted at 43,964
additional transitions; its policy, optimizer and matching VecNormalize were
saved as interrupted.zip/interrupted_normalize.pkl. The failed status records
KeyboardInterrupt, not a model failure. No prior results or frozen config edited.

V007 uses CUDA on the verified GTX 1070 Ti, eight SubprocVecEnv physics workers,
and batch size 1,024. It resumes that exact saved pair and requests the remaining
156,036 transitions, bounded by 1,200 training seconds plus evaluations. Launch
verified a CUDA matrix multiplication and loaded the transfer pair. Initial
paired evaluation runs before learning, so the monitor initially reports zero.

PufferLib assessment: current 5.0 documentation describes a native C/CUDA trainer
and C environment API, not a drop-in Python/Gymnasium backend. Its subsecond
installation example is Breakout on RTX 5090. Our MuJoCo environment would require
a port preserving tether dynamics, contact latching, domain randomization,
normalization and held-out evaluation. Seconds-scale training for this task is
not established. Reference: https://puffer.ai/docs.html

The native .NET app was rebuilt and reopened. Its training button now starts
V007 CUDA continuations. Monitor unit tests: 2 passed. CUDA end-to-end throughput
and held-out improvement remain pending; no speedup claim yet.
