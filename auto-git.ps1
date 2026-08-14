# SARV Auto Git
# Automatically detects changes, commits them, and pushes to origin/main.

$Repo = "C:\Users\samak\Sarv"
$Branch = "main"
$Remote = "origin"

Set-Location $Repo

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "       SARV AUTO GIT WATCHER" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Repository: $Repo"
Write-Host "Branch:     $Branch"
Write-Host "Remote:     $Remote"
Write-Host ""
Write-Host "Watching for changes..."
Write-Host "Press Ctrl+C to stop."
Write-Host ""

function Test-GitOperationInProgress {

    $gitDir = (git rev-parse --git-dir 2>$null)

    if (-not $gitDir) {
        return $true
    }

    if (
        (Test-Path (Join-Path $gitDir "rebase-merge")) -or
        (Test-Path (Join-Path $gitDir "rebase-apply")) -or
        (Test-Path (Join-Path $gitDir "MERGE_HEAD")) -or
        (Test-Path (Join-Path $gitDir "CHERRY_PICK_HEAD"))
    ) {
        return $true
    }

    return $false
}

function Commit-And-Push {

    Write-Host ""
    Write-Host "------------------------------------------" -ForegroundColor DarkGray
    Write-Host "Change detected." -ForegroundColor Yellow

    if (Test-GitOperationInProgress) {
        Write-Host "Git operation currently in progress." -ForegroundColor Red
        Write-Host "Auto-Git will NOT interfere with it." -ForegroundColor Red
        return
    }

    git fetch $Remote 2>$null

    if ($LASTEXITCODE -ne 0) {
        Write-Host "git fetch failed. Skipping this cycle." -ForegroundColor Red
        return
    }

    $status = git status --porcelain

    if (-not $status) {
        return
    }

    Write-Host ""
    Write-Host "Changes:" -ForegroundColor Cyan
    git status --short

    git add -A

    if ($LASTEXITCODE -ne 0) {
        Write-Host "git add failed." -ForegroundColor Red
        return
    }

    $staged = git diff --cached --name-only

    if (-not $staged) {
        Write-Host "Nothing staged." -ForegroundColor Yellow
        return
    }

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

    $commitMessage = "auto: update SARV - $timestamp"

    Write-Host ""
    Write-Host "Committing..." -ForegroundColor Cyan

    git commit -m $commitMessage

    if ($LASTEXITCODE -ne 0) {
        Write-Host "Commit failed." -ForegroundColor Red
        return
    }

    Write-Host ""
    Write-Host "Pushing to GitHub..." -ForegroundColor Cyan

    git push $Remote $Branch

    if ($LASTEXITCODE -eq 0) {
        Write-Host ""
        Write-Host "SUCCESS: committed and pushed." -ForegroundColor Green
        Write-Host "Commit: $commitMessage" -ForegroundColor Green
    }
    else {
        Write-Host ""
        Write-Host "Push failed." -ForegroundColor Red
        Write-Host "The local commit is preserved." -ForegroundColor Yellow
        Write-Host "Auto-Git will not attempt a dangerous force push." -ForegroundColor Yellow
    }

    Write-Host "------------------------------------------"
}

# Initial check
Write-Host "Checking repository..." -ForegroundColor Cyan

if (Test-GitOperationInProgress) {
    Write-Host ""
    Write-Host "WARNING: A Git operation is currently in progress." -ForegroundColor Yellow
    Write-Host "Auto-Git will wait until it is finished." -ForegroundColor Yellow
}

# Watch repository
$lastSnapshot = ""

while ($true) {

    if (-not (Test-GitOperationInProgress)) {

        $currentSnapshot = (
            git status --porcelain 2>$null |
            Out-String
        ).Trim()

        if ($currentSnapshot -ne $lastSnapshot) {

            $lastSnapshot = $currentSnapshot

            if ($currentSnapshot) {
                Commit-And-Push
            }
        }
    }

    Start-Sleep -Seconds 5
}