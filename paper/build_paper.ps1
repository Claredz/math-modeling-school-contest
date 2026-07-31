$ErrorActionPreference = 'Stop'
$PaperRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $PaperRoot

$latexmk = Get-Command latexmk -ErrorAction SilentlyContinue
if ($null -ne $latexmk) {
    & $latexmk.Source -xelatex -interaction=nonstopmode -halt-on-error main.tex
    if ($LASTEXITCODE -ne 0) { throw "latexmk failed with exit code $LASTEXITCODE" }
} else {
    $xelatex = Get-Command xelatex -ErrorAction SilentlyContinue
    if ($null -eq $xelatex) { throw 'Neither latexmk nor xelatex is available.' }
    & $xelatex.Source -interaction=nonstopmode -halt-on-error main.tex
    if ($LASTEXITCODE -ne 0) { throw "first xelatex failed with exit code $LASTEXITCODE" }
    & bibtex main
    if ($LASTEXITCODE -ne 0) { throw "bibtex failed with exit code $LASTEXITCODE" }
    & $xelatex.Source -interaction=nonstopmode -halt-on-error main.tex
    if ($LASTEXITCODE -ne 0) { throw "second xelatex failed with exit code $LASTEXITCODE" }
    & $xelatex.Source -interaction=nonstopmode -halt-on-error main.tex
    if ($LASTEXITCODE -ne 0) { throw "third xelatex failed with exit code $LASTEXITCODE" }
}

if (-not (Test-Path main.pdf)) { throw 'main.pdf was not produced.' }
$pdf = Get-Item main.pdf
Write-Output ("Built {0}; bytes={1}; absolute_path={2}" -f $pdf.Name, $pdf.Length, $pdf.FullName)
