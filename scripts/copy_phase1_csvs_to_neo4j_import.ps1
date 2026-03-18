param(
  [Parameter(Mandatory=$true)]
  [string]$ImportDir,

  [string]$Phase1OutputDir = ".\phase1_output_v2"
)

$ImportDir = (Resolve-Path $ImportDir).Path
$Phase1OutputDir = (Resolve-Path $Phase1OutputDir).Path

Write-Host "Copying Phase-1 v2 CSVs to Neo4j import directory..." -ForegroundColor Cyan
Write-Host "  From: $Phase1OutputDir"
Write-Host "  To:   $ImportDir"

# All 17 v2 CSV files required by 02_load_nodes_v2.cypher and 03_load_edges_v2.cypher
$files = @(
  # Node files
  "acts.csv",
  "parts.csv",
  "chapters.csv",
  "sections.csv",
  "articles.csv",
  "definitions.csv",
  "cases_sc_neo4j.csv",
  "cases_iltur_neo4j.csv",
  # Edge files
  "act_part.csv",
  "part_chapter.csv",
  "chapter_section.csv",
  "act_section.csv",
  "act_article.csv",
  "section_defines_term.csv",
  "section_references_section.csv",
  "case_cites_section.csv",
  "case_cites_article.csv"
)

$copied = 0
$missing = @()

foreach ($f in $files) {
  $src = Join-Path $Phase1OutputDir $f
  if (!(Test-Path $src)) {
    $missing += $f
    Write-Warning "  Missing: $f (will skip)"
    continue
  }
  $dest = Join-Path $ImportDir $f
  Copy-Item -Path $src -Destination $dest -Force
  $size = (Get-Item $src).Length
  Write-Host "  Copied: $f ($size bytes)" -ForegroundColor Green
  $copied++
}

Write-Host ""
Write-Host "Done. Copied $copied / $($files.Count) files." -ForegroundColor Cyan

if ($missing.Count -gt 0) {
  Write-Host "Missing files (not copied):" -ForegroundColor Yellow
  foreach ($f in $missing) {
    Write-Host "  - $f" -ForegroundColor Yellow
  }
  exit 1
}
