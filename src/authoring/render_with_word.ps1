param([Parameter(Mandatory=$true)][string]$source,[Parameter(Mandatory=$true)][string]$renderDir)
$ErrorActionPreference = 'Stop'
$pdf = Join-Path $renderDir 'report.pdf'
New-Item -ItemType Directory -Path $renderDir -Force | Out-Null
$previousIds = @(Get-Process WINWORD -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id)
$word = $null
$wordDoc = $null
$isolated = $false
try {
    $word = New-Object -ComObject Word.Application
    $newIds = @(Get-Process WINWORD -ErrorAction SilentlyContinue | Where-Object { $_.Id -notin $previousIds } | Select-Object -ExpandProperty Id)
    if ($newIds.Count -ne 1 -or $word.Documents.Count -ne 0) {
        throw 'An isolated Word instance was not available; no user document was changed.'
    }
    $isolated = $true
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.AutomationSecurity = 3
    $wordDoc = $word.Documents.Open($source, $false, $true, $false)
    $wordDoc.Repaginate()
    $wordDoc.Fields.Update() | Out-Null
    $wordDoc.ExportAsFixedFormat($pdf, 17)
    @{PDF=$pdf; Pages=$wordDoc.ComputeStatistics(2); IsolatedProcess=$newIds[0]} | ConvertTo-Json -Compress
} finally {
    if ($null -ne $wordDoc) {
        $wordDoc.Close(0)
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($wordDoc)
    }
    if ($null -ne $word) {
        if ($isolated) { $word.Quit(0) }
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($word)
    }
}
if (-not (Test-Path -LiteralPath $pdf)) { throw 'PDF rendering did not finish' }
& 'C:\Users\ACER\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe' -r 115 -png $pdf (Join-Path $renderDir 'page')
if ($LASTEXITCODE -ne 0) { throw 'Page image rendering failed' }
