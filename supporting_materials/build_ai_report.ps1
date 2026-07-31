$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$source = Get-ChildItem -LiteralPath $Root -File -Filter '*.tex' |
    Where-Object { $_.BaseName -ne '__ai_usage_build' } |
    Select-Object -First 1
if ($null -eq $source) { throw 'No AI usage .tex source was found.' }
$tempBase = '__ai_usage_build'
$tempTex = Join-Path $Root ($tempBase + '.tex')
Copy-Item -LiteralPath $source.FullName -Destination $tempTex -Force
& xelatex -interaction=nonstopmode -halt-on-error ($tempBase + '.tex')
if ($LASTEXITCODE -ne 0) { throw "first xelatex failed with exit code $LASTEXITCODE" }
& xelatex -interaction=nonstopmode -halt-on-error ($tempBase + '.tex')
if ($LASTEXITCODE -ne 0) { throw "second xelatex failed with exit code $LASTEXITCODE" }
$built = Get-Item ($tempBase + '.pdf')
$output = Join-Path $Root ($source.BaseName + '.pdf')
Move-Item -LiteralPath $built.FullName -Destination $output -Force
Get-ChildItem -LiteralPath $Root -File -Filter ($tempBase + '.*') | Remove-Item -Force
$pdf = Get-Item -LiteralPath $output
Write-Output ("Built {0}; bytes={1}; absolute_path={2}" -f $pdf.Name, $pdf.Length, $pdf.FullName)
