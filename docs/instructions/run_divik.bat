@echo off
tasklist /FI "IMAGENAME eq Docker Desktop.exe" 2>NUL | find /I /N "Docker Desktop.exe">NUL
if "%ERRORLEVEL%"=="0" (
    echo Docker is running
) ELSE (
    echo Docker is not running, launching - please wait....
    start "" /B "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    timeout 60 /nobreak
)
echo Checking for updates...
docker pull gmrukwa/divik
docker run^
    --rm^
    -it^
    --volume %cd%:/data^
    gmrukwa/divik^
    divik^
    --source /data/data.csv^
    --config /data/divik.json^
    --destination /data/results^
    --verbose
pause
