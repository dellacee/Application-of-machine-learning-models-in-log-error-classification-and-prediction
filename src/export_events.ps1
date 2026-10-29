# Xuất Windows Event Log (Application + System) ra CSV để làm dữ liệu huấn luyện.
# Chạy trong PowerShell thường (không cần quyền admin):  .\src\export_events.ps1
$out = Join-Path $PSScriptRoot "..\data\events_raw.csv"
Get-WinEvent -LogName Application, System -MaxEvents 200000 -ErrorAction SilentlyContinue |
    Select-Object TimeCreated, Level, LevelDisplayName, Id, ProviderName, LogName, Message |
    Export-Csv $out -NoTypeInformation -Encoding UTF8
Write-Host "Đã xuất: $out"
