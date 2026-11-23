<#
 Deploy the malaria risk service to AWS (Lambda container + API Gateway) (Windows PowerShell).
 Prerequisites: AWS CLI v2 configured (aws configure / SSO), Docker Desktop running, models/ and app/cluster_scores.json present
 (run `python -m src.train_final` first). Run from the repository root:   .\deploy\deploy.ps1 -Region eu-west-1
#>
param([string]$Region = "eu-west-1", [string]$Stack = "malaria-risk", [string]$Service = "malaria-risk")
$ErrorActionPreference = "Stop"
function Need($cmd) { if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) { throw "$cmd is required but was not found" } }
Need aws; Need docker
foreach ($p in "models/malaria_risk_model.joblib", "models/model_card.json", "app/cluster_scores.json") { if (-not (Test-Path $p)) { throw "Missing $p. Run: python -m src.train_final" } }

$acct = aws sts get-caller-identity --query Account --output text --region $Region
Write-Host "AWS account $acct, region $Region"
$tag = (Get-Date -Format "yyyyMMdd-HHmmss")
$registry = "$acct.dkr.ecr.$Region.amazonaws.com"

Write-Host "Stage 1: ECR repository"
aws cloudformation deploy --region $Region --stack-name $Stack --template-file deploy/cloudformation.yaml `
  --capabilities CAPABILITY_IAM --parameter-overrides ServiceName=$Service CreateService=false --no-fail-on-empty-changeset

Write-Host "Build and push image $Service`:$tag"
aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin $registry
docker build --platform linux/amd64 --provenance=false -t "${registry}/${Service}:${tag}" .
docker push "${registry}/${Service}:${tag}"

Write-Host "Stage 2: Lambda function + API Gateway"
aws cloudformation deploy --region $Region --stack-name $Stack --template-file deploy/cloudformation.yaml `
  --capabilities CAPABILITY_IAM --parameter-overrides ServiceName=$Service CreateService=true ImageTag=$tag --no-fail-on-empty-changeset

$url = aws cloudformation describe-stacks --region $Region --stack-name $Stack --query "Stacks[0].Outputs[?OutputKey=='ServiceUrl'].OutputValue" --output text
Write-Host "Service URL: $url"
$r = Invoke-WebRequest "$url/health" -UseBasicParsing
Write-Host "Health check: $($r.StatusCode) $($r.Content)"
