param([switch]$DryRun)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$modelDir = Join-Path $root 'models\parakeet-tdt-0.6b-v2-ov'
$modelId = 'FluidInference/parakeet-tdt-0.6b-v2-ov'
$revision = 'e115adf711921df9378e383d01190901f0057704'
$sherpaDir = Join-Path $root 'models\sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8'
$sherpaId = 'csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8'
$sherpaRevision = '1ab9323565ddb038682214b292f588070a538ce2'
$vadPath = Join-Path $root 'models\silero_vad.onnx'
$vadUrl = 'https://raw.githubusercontent.com/snakers4/silero-vad/1e261b036686cd0017d500ee96acd1c4ba572a9d/src/silero_vad/data/silero_vad.onnx'
$python = Join-Path $root '.venv\Scripts\python.exe'

if (-not (Test-Path $python)) {
    throw 'Run .\setup.ps1 first to install the pinned huggingface-hub package.'
}

foreach ($download in @(
    @{ Id = $modelId; Revision = $revision; Destination = $modelDir },
    @{ Id = $sherpaId; Revision = $sherpaRevision; Destination = $sherpaDir }
)) {
    $downloadArgs = @(
        'download', $download.Id,
        '--revision', $download.Revision,
        '--local-dir', $download.Destination
    )
    if ($DryRun) { $downloadArgs += '--dry-run' }
    & $python -c "from huggingface_hub.cli.hf import main; main()" @downloadArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Hugging Face download failed for $($download.Id); see output above."
    }
}

if ($DryRun) {
    $head = Invoke-WebRequest -Uri $vadUrl -Method Head
    Write-Host "VAD source reachable: $vadUrl (HTTP $([int]$head.StatusCode))"
} else {
    Invoke-WebRequest -Uri $vadUrl -OutFile $vadPath
}

$required = @(
    'parakeet_melspectogram.xml', 'parakeet_melspectogram.bin',
    'parakeet_encoder.xml', 'parakeet_encoder.bin',
    'parakeet_decoder.xml', 'parakeet_decoder.bin',
    'parakeet_joint.xml', 'parakeet_joint.bin', 'parakeet_vocab.json'
)
if (-not $DryRun) {
    foreach ($name in $required) {
        if (-not (Test-Path (Join-Path $modelDir $name))) {
            throw "Downloaded model snapshot is missing $name."
        }
    }
    foreach ($name in @('encoder.int8.onnx', 'decoder.int8.onnx', 'joiner.int8.onnx', 'tokens.txt')) {
        if (-not (Test-Path (Join-Path $sherpaDir $name))) {
            throw "Downloaded sherpa-onnx model snapshot is missing $name."
        }
    }
    if (-not (Test-Path $vadPath)) {
        throw 'Silero VAD download is missing.'
    }
}

if ($DryRun) {
    Write-Host 'Pinned Parakeet, sherpa-onnx, and Silero VAD sources are reachable; no model weights were downloaded.'
} else {
    Write-Host "Downloaded pinned Parakeet OpenVINO, sherpa-onnx CPU fallback, and Silero VAD assets under $root\models"
}
