# Opt-in failure-priority termination correction

A physical zone failure on step500 previously also had truncated=True and was omitted by done & ~truncated crash counting. termination_version2 now gives physical failure priority:truncated=timeout & (failed==0),done=failed ortruncated. Version1 preserved bydefault and all frozen duration/mask factorial plans, so this correction is a separate factor. No effect on current250stepeval or recoverydiagnostic with timeout disabled.

Actual production CudaVector.send under no-op CPUphysics mixedbatch test verifies version1omits overlap,version2counts realfailure,retains truetimeout,terminalreward-1 andpostterminalreset. Sources/scripts/check_failure_priority.py and deep_audit/root/failure_priority_production_test.json. No synthetic helper controls or GPUcompile. Syntax passed.

Test-harness audit correction: initial independent reset factory substitution missed torch.rand with CUDA device, although physical engine was CPU/no-op. Therefore that first test was not strictlyCPU-only and is retained with its originalscope caveat. Rootrerun substitutes rand too and disables CUDA_VISIBLE_DEVICES; resetidentity/overlap findings reproduce exactly. No duplicate trainer was created; subsequent tests require hiddenGPU to enforce CPU scope. StrictCPUrerun source/results preserved separately.
