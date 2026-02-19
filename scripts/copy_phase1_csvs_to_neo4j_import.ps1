param(
  [Parameter(Mandatory=$true)]
  [string]$ImportDir,

  [string]$Phase1OutputDir = ".\\phase1_output"
)

$ImportDir = (Resolve-Path $ImportDir).Path
$Phase1OutputDir = (Resolve-Path $Phase1OutputDir).Path

Write-Host "Copying Phase-1 CSVs..." -ForegroundColor Cyan
Write-Host "  From: $Phase1OutputDir"
Write-Host "  To:   $ImportDir"

$files = @("acts.csv","sections.csv","articles.csv","cases.csv","edges.csv")

foreach ($f in $files) {
  $src = Join-Path $Phase1OutputDir $f
  if (!(Test-Path $src)) {
    throw "Missing file: $src"
  }
  Copy-Item -Path $src -Destination (Join-Path $ImportDir $f) -Force
  Write-Host "  Copied: $f"
}

Write-Host "Done." -ForegroundColor Green

