[CmdletBinding()]
param(
    [ValidateSet("Train", "Deploy", "Status", "Logs", "Pull", "StopTraining", "StopPod", "PodStatus", "Help")]
    [string]$Action = "Help",
    [string]$Config = "configs/ai_gp_001_gpu_ppo.yaml",
    [string]$Experiment,
    [string]$HostName,
    [int]$Port = 0,
    [string]$KeyPath,
    [switch]$AutoStopPod
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$RemoteRoot = "/root/drone-rl-lab"
$PodIdFile = if ($env:DRONE_RL_RUNPOD_POD_ID_FILE) {
    $env:DRONE_RL_RUNPOD_POD_ID_FILE
} else {
    Join-Path $HOME ".config\drone-rl-lab\runpod_pod_id"
}

function Show-Usage {
    Write-Host @"
AI-GP RunPod workflow

Required once:
  `$env:RUNPOD_API_KEY = "..."
  `$env:RUNPOD_POD_ID = "..."
  `$env:DRONE_RL_DEPLOY_KEY = "`$HOME\.ssh\id_ed25519_runpod"

Commands:
  .\scripts\runpod_ai_gp.ps1 Train
  .\scripts\runpod_ai_gp.ps1 Status -Experiment ai_gp_001_gpu_ppo
  .\scripts\runpod_ai_gp.ps1 Logs -Experiment ai_gp_001_gpu_ppo
  .\scripts\runpod_ai_gp.ps1 Pull -Experiment ai_gp_001_gpu_ppo
  .\scripts\runpod_ai_gp.ps1 StopTraining -Experiment ai_gp_001_gpu_ppo
  .\scripts\runpod_ai_gp.ps1 StopPod

Use -HostName and -Port to bypass RunPod API endpoint discovery.
"@
}

function Get-ExperimentName {
    param([string]$ConfigPath)
    if ($Experiment) {
        return $Experiment
    }
    $line = Get-Content -LiteralPath $ConfigPath |
        Where-Object { $_ -match "^\s*name:\s*(.+?)\s*$" } |
        Select-Object -First 1
    if (-not $line) {
        throw "Could not read experiment name from $ConfigPath"
    }
    $name = ([regex]::Match($line, "^\s*name:\s*(.+?)\s*$")).Groups[1].Value.Trim("'`" ")
    if ($name -notmatch "^[A-Za-z0-9_.-]+$") {
        throw "Unsafe experiment name: $name"
    }
    return $name
}

function Resolve-KeyPath {
    if ($KeyPath) {
        return (Resolve-Path -LiteralPath $KeyPath).Path
    }
    $candidates = @(
        $env:DRONE_RL_DEPLOY_KEY,
        (Join-Path $HOME ".ssh\id_ed25519_runpod"),
        (Join-Path $HOME ".ssh\id_ed25519")
    ) | Where-Object { $_ }
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    throw "No SSH key found. Set DRONE_RL_DEPLOY_KEY or pass -KeyPath."
}

function Get-PodId {
    if ($env:RUNPOD_POD_ID) {
        return $env:RUNPOD_POD_ID
    }
    if (Test-Path -LiteralPath $PodIdFile) {
        return (Get-Content -LiteralPath $PodIdFile -Raw).Trim()
    }
    throw "RUNPOD_POD_ID is not set and $PodIdFile does not exist."
}

function Invoke-RunPodQuery {
    param([string]$Query)
    if (-not $env:RUNPOD_API_KEY) {
        throw "RUNPOD_API_KEY is required for pod control and endpoint discovery."
    }
    $headers = @{ Authorization = "Bearer $($env:RUNPOD_API_KEY)" }
    $body = @{ query = $Query } | ConvertTo-Json -Compress
    return Invoke-RestMethod `
        -Uri "https://api.runpod.io/graphql" `
        -Method Post `
        -Headers $headers `
        -ContentType "application/json" `
        -Body $body
}

function Get-Pod {
    $podId = Get-PodId
    $response = Invoke-RunPodQuery "{ pod(input: {podId: `"$podId`"}) { id desiredStatus runtime { ports { ip privatePort publicPort type } } } }"
    if (-not $response.data.pod) {
        throw "RunPod pod not found: $podId"
    }
    return $response.data.pod
}

function Start-Pod {
    $pod = Get-Pod
    if ($pod.desiredStatus -ne "RUNNING") {
        Write-Host "[runpod] Resuming pod $($pod.id)"
        [void](Invoke-RunPodQuery "mutation { podResume(input: {podId: `"$($pod.id)`", gpuCount: 1}) { id desiredStatus } }")
    }
    $deadline = (Get-Date).AddMinutes(5)
    do {
        $pod = Get-Pod
        if ($pod.desiredStatus -eq "RUNNING" -and $pod.runtime) {
            return $pod
        }
        Start-Sleep -Seconds 5
    } while ((Get-Date) -lt $deadline)
    throw "Pod did not become ready within five minutes."
}

function Resolve-Endpoint {
    if ($HostName -and $Port -gt 0) {
        return @{ Host = $HostName; Port = $Port }
    }
    $pod = Get-Pod
    foreach ($entry in $pod.runtime.ports) {
        if ($entry.privatePort -eq 22 -and $entry.type -eq "tcp") {
            return @{ Host = [string]$entry.ip; Port = [int]$entry.publicPort }
        }
    }
    throw "Pod has no public SSH endpoint."
}

function Initialize-Ssh {
    $script:ResolvedKey = Resolve-KeyPath
    $endpoint = Resolve-Endpoint
    $script:ResolvedHost = $endpoint.Host
    $script:ResolvedPort = $endpoint.Port
    $script:SshBase = @(
        "-o", "StrictHostKeyChecking=no",
        "-o", "BatchMode=yes",
        "-i", $script:ResolvedKey,
        "-p", [string]$script:ResolvedPort
    )
}

function Wait-Ssh {
    $deadline = (Get-Date).AddMinutes(2)
    do {
        & ssh.exe @script:SshBase "root@$script:ResolvedHost" "echo ready" 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) {
            return
        }
        Start-Sleep -Seconds 4
    } while ((Get-Date) -lt $deadline)
    throw "SSH did not become ready."
}

function Invoke-Ssh {
    param([string]$Command)
    & ssh.exe @script:SshBase "root@$script:ResolvedHost" $Command
    if ($LASTEXITCODE -ne 0) {
        throw "SSH command failed with exit code $LASTEXITCODE"
    }
}

function Deploy-Code {
    param([string]$ResolvedConfig)
    $relativeConfig = $ResolvedConfig.Substring($RepoRoot.Length).TrimStart("\", "/").Replace("\", "/")
    $archive = Join-Path ([IO.Path]::GetTempPath()) "drone-rl-ai-gp-$([guid]::NewGuid().ToString('N')).tar.gz"
    try {
        Push-Location $RepoRoot
        & tar.exe -czf $archive `
            "--exclude=__pycache__" `
            "--exclude=*.pyc" `
            "ai_gp_rl" `
            "train.py" `
            "train_ai_gp.py" `
            "scripts/smoke_ai_gp.py" `
            "scripts/runpod_ai_gp_remote.sh" `
            $relativeConfig
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to build deployment archive."
        }
        Pop-Location

        Write-Host "[deploy] Uploading minimal AI-GP training bundle"
        $scpArgs = @(
            "-o", "StrictHostKeyChecking=no",
            "-i", $script:ResolvedKey,
            "-P", [string]$script:ResolvedPort,
            $archive,
            "root@$script:ResolvedHost`:/root/drone-rl-ai-gp-upload.tar.gz"
        )
        & scp.exe @scpArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Upload failed."
        }
        Invoke-Ssh "mkdir -p $RemoteRoot && tar -xzf /root/drone-rl-ai-gp-upload.tar.gz -C $RemoteRoot && rm -f /root/drone-rl-ai-gp-upload.tar.gz"
        Invoke-Ssh "bash $RemoteRoot/scripts/runpod_ai_gp_remote.sh bootstrap"
        Invoke-Ssh "bash $RemoteRoot/scripts/runpod_ai_gp_remote.sh smoke '$relativeConfig'"
    } finally {
        if ((Get-Location).Path -ne $RepoRoot) {
            Pop-Location
        }
        Remove-Item -LiteralPath $archive -Force -ErrorAction SilentlyContinue
    }
    return $relativeConfig
}

if ($Action -eq "Help") {
    Show-Usage
    exit 0
}

$configCandidate = if ([IO.Path]::IsPathRooted($Config)) {
    $Config
} else {
    Join-Path $RepoRoot $Config
}
$resolvedConfig = (Resolve-Path -LiteralPath $configCandidate).Path
$experimentName = Get-ExperimentName $resolvedConfig

if ($Action -eq "StopPod") {
    $podId = Get-PodId
    [void](Invoke-RunPodQuery "mutation { podStop(input: {podId: `"$podId`"}) { id desiredStatus } }")
    Write-Host "[runpod] Stop requested for $podId"
    exit 0
}

if ($Action -eq "PodStatus") {
    $pod = Get-Pod
    Write-Host "pod=$($pod.id) status=$($pod.desiredStatus)"
    exit 0
}

if ($Action -in @("Train", "Deploy") -and -not ($HostName -and $Port -gt 0)) {
    [void](Start-Pod)
}

Initialize-Ssh
Wait-Ssh

switch ($Action) {
    "Deploy" {
        [void](Deploy-Code $resolvedConfig)
        Write-Host "[deploy] Ready at $RemoteRoot"
    }
    "Train" {
        $relativeConfig = Deploy-Code $resolvedConfig
        $autoStop = if ($AutoStopPod) { "true" } else { "false" }
        Invoke-Ssh "bash $RemoteRoot/scripts/runpod_ai_gp_remote.sh start '$relativeConfig' '$experimentName' '$autoStop'"
        Write-Host "[train] Follow with: .\scripts\runpod_ai_gp.ps1 Logs -Experiment $experimentName"
    }
    "Status" {
        Invoke-Ssh "bash $RemoteRoot/scripts/runpod_ai_gp_remote.sh status '$experimentName'"
    }
    "Logs" {
        $logArgs = @("-t") + $script:SshBase + @(
            "root@$script:ResolvedHost",
            "bash $RemoteRoot/scripts/runpod_ai_gp_remote.sh logs '$experimentName'"
        )
        & ssh.exe @logArgs
    }
    "Pull" {
        $localResults = Join-Path $RepoRoot "results"
        New-Item -ItemType Directory -Force -Path $localResults | Out-Null
        $scpArgs = @(
            "-r",
            "-o", "StrictHostKeyChecking=no",
            "-i", $script:ResolvedKey,
            "-P", [string]$script:ResolvedPort,
            "root@$script:ResolvedHost`:$RemoteRoot/results/$experimentName",
            $localResults
        )
        & scp.exe @scpArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Result download failed."
        }
        Write-Host "[pull] Results copied to $localResults\$experimentName"
    }
    "StopTraining" {
        Invoke-Ssh "bash $RemoteRoot/scripts/runpod_ai_gp_remote.sh stop '$experimentName'"
    }
}
