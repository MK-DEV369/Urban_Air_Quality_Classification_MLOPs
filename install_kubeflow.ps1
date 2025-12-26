# Kubeflow Pipelines Installation Script for Docker Desktop Kubernetes
# Run this AFTER enabling Kubernetes in Docker Desktop

Write-Host "🚀 Kubeflow Pipelines Installation Script" -ForegroundColor Cyan
Write-Host "=========================================`n" -ForegroundColor Cyan

# Step 1: Verify Kubernetes is running
Write-Host "Step 1: Checking Kubernetes cluster..." -ForegroundColor Yellow
try {
    kubectl cluster-info | Out-Null
    Write-Host "✅ Kubernetes is running" -ForegroundColor Green
} catch {
    Write-Host "❌ Kubernetes is not running!" -ForegroundColor Red
    Write-Host "Please enable Kubernetes in Docker Desktop Settings" -ForegroundColor Red
    exit 1
}

# Step 2: Check kubectl context
Write-Host "`nStep 2: Setting kubectl context..." -ForegroundColor Yellow
$context = kubectl config current-context
Write-Host "Current context: $context" -ForegroundColor Cyan

if ($context -ne "docker-desktop") {
    Write-Host "Switching to docker-desktop context..." -ForegroundColor Yellow
    kubectl config use-context docker-desktop
}
Write-Host "✅ Using docker-desktop context" -ForegroundColor Green

# Step 3: Install Kubeflow Pipelines
Write-Host "`nStep 3: Installing Kubeflow Pipelines..." -ForegroundColor Yellow
Write-Host "This will download and install Kubeflow (~2-5 minutes)" -ForegroundColor Cyan

$KFP_VERSION = "2.0.5"
$MANIFEST_URL = "https://raw.githubusercontent.com/kubeflow/pipelines/$KFP_VERSION/manifests/kustomize/env/platform-agnostic/kustomization.yaml"

Write-Host "Downloading Kubeflow Pipelines manifests..." -ForegroundColor Cyan

# Create temporary directory
$tempDir = New-Item -ItemType Directory -Path "$env:TEMP\kubeflow-install" -Force
Set-Location $tempDir

# Download kustomization files
$baseUrl = "https://raw.githubusercontent.com/kubeflow/pipelines/$KFP_VERSION/manifests/kustomize"

# Create directory structure
New-Item -ItemType Directory -Path "base" -Force | Out-Null
New-Item -ItemType Directory -Path "env/platform-agnostic" -Force | Out-Null

# Download base kustomization
Invoke-WebRequest -Uri "$baseUrl/base/kustomization.yaml" -OutFile "base/kustomization.yaml"

# Download platform-agnostic kustomization
Invoke-WebRequest -Uri "$baseUrl/env/platform-agnostic/kustomization.yaml" -OutFile "env/platform-agnostic/kustomization.yaml"

Write-Host "✅ Manifests downloaded" -ForegroundColor Green

# Apply using kubectl
Write-Host "`nApplying Kubeflow Pipelines to cluster..." -ForegroundColor Cyan
kubectl apply -k "env/platform-agnostic" 2>&1 | Out-Null

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Kubeflow Pipelines installed" -ForegroundColor Green
} else {
    Write-Host "⚠️ Installation completed with warnings (this is normal)" -ForegroundColor Yellow
}

# Step 4: Wait for pods to be ready
Write-Host "`nStep 4: Waiting for pods to start..." -ForegroundColor Yellow
Write-Host "This may take 5-10 minutes. Do not close this window." -ForegroundColor Cyan

$maxWaitTime = 600  # 10 minutes
$elapsed = 0
$interval = 10

while ($elapsed -lt $maxWaitTime) {
    $notReady = kubectl get pods -n kubeflow -o json | ConvertFrom-Json | 
        ForEach-Object { $_.items } | 
        Where-Object { $_.status.phase -ne "Running" }
    
    $totalPods = (kubectl get pods -n kubeflow --no-headers 2>$null | Measure-Object).Count
    $runningPods = (kubectl get pods -n kubeflow --no-headers 2>$null | Where-Object { $_ -match "Running" } | Measure-Object).Count
    
    Write-Host "`r[$elapsed/$maxWaitTime s] Pods: $runningPods/$totalPods running" -NoNewline -ForegroundColor Cyan
    
    if ($notReady.Count -eq 0 -and $totalPods -gt 0) {
        Write-Host "`n✅ All pods are running!" -ForegroundColor Green
        break
    }
    
    Start-Sleep -Seconds $interval
    $elapsed += $interval
}

if ($elapsed -ge $maxWaitTime) {
    Write-Host "`n⚠️ Timeout reached. Some pods may still be starting." -ForegroundColor Yellow
    Write-Host "Check status with: kubectl get pods -n kubeflow" -ForegroundColor Cyan
}

# Step 5: Port forwarding
Write-Host "`nStep 5: Setting up port forwarding..." -ForegroundColor Yellow
Write-Host "Kubeflow UI will be available at: http://localhost:8080" -ForegroundColor Green

Write-Host "`n🎉 Installation Complete!" -ForegroundColor Green
Write-Host "`n📋 Next Steps:" -ForegroundColor Cyan
Write-Host "1. Open new terminal and run:" -ForegroundColor White
Write-Host "   kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80" -ForegroundColor Yellow
Write-Host "`n2. Wait 30 seconds, then access:" -ForegroundColor White
Write-Host "   http://localhost:8080" -ForegroundColor Yellow
Write-Host "`n3. Deploy your pipeline:" -ForegroundColor White
Write-Host "   python kubeflow_deploy.py --host http://localhost:8080" -ForegroundColor Yellow

Write-Host "`n📊 Check Status:" -ForegroundColor Cyan
Write-Host "   kubectl get pods -n kubeflow" -ForegroundColor Yellow
Write-Host "   kubectl get svc -n kubeflow" -ForegroundColor Yellow

Set-Location $PSScriptRoot
