param(
    [string]$EnvironmentName = 'oai_multi_agent_system',
    [int]$Port = 7860
)

$ErrorActionPreference = 'Stop'
$pythonPath = Join-Path $PSScriptRoot "$EnvironmentName\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "The $EnvironmentName environment is missing."
}
$mysqlRuntime = Join-Path $PSScriptRoot 'oai_multi_agent_system\mysql_runtime'
$databaseConfigPath = Join-Path $PSScriptRoot 'src\db_utils\db_config.json'
if ((Test-Path -LiteralPath (Join-Path $mysqlRuntime 'data')) -and
    (Test-Path -LiteralPath $databaseConfigPath)) {
    $databaseConfig = Get-Content -LiteralPath $databaseConfigPath -Raw | ConvertFrom-Json
    if ($databaseConfig.host -eq '127.0.0.1' -and $databaseConfig.port -eq 3306) {
        $probe = [System.Net.Sockets.TcpClient]::new()
        try { $probe.Connect('127.0.0.1', 3306) } catch { }
        $databaseReady = $probe.Connected
        $probe.Dispose()
        if (-not $databaseReady) {
            $mysqlPath = 'C:\Program Files\MySQL\MySQL Server 8.4\bin\mysqld.exe'
            $databaseProcess = Start-Process -FilePath $mysqlPath -ArgumentList @(
                '--no-defaults', "--datadir=`"$(Join-Path $mysqlRuntime 'data')`"",
                '--bind-address=127.0.0.1', '--port=3306', '--mysqlx=OFF', '--console'
            ) -WindowStyle Hidden -RedirectStandardOutput (Join-Path $mysqlRuntime 'stdout.log') `
                -RedirectStandardError (Join-Path $mysqlRuntime 'stderr.log') -PassThru
            for ($attempt = 0; $attempt -lt 30; $attempt++) {
                $probe = [System.Net.Sockets.TcpClient]::new()
                try { $probe.Connect('127.0.0.1', 3306) } catch { }
                $databaseReady = $probe.Connected
                $probe.Dispose()
                if ($databaseReady) { break }
                if ($databaseProcess.HasExited) { throw 'Local MySQL exited. Check mysql_runtime logs.' }
                Start-Sleep -Seconds 1
            }
            if (-not $databaseReady) { throw 'Local MySQL startup timed out.' }
        }
    }
}
Push-Location (Join-Path $PSScriptRoot 'src')
try {
    & $pythonPath -m uvicorn app:app --host 127.0.0.1 --port $Port
    if ($LASTEXITCODE -ne 0) {
        throw "Application exited with code $LASTEXITCODE."
    }
} finally {
    Pop-Location
}
