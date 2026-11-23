# Remove everything created by deploy.ps1 (Lambda function, API, log group, role, ECR repository and its images).
param([string]$Region = "eu-west-1", [string]$Stack = "malaria-risk")
$ErrorActionPreference = "Stop"
aws cloudformation delete-stack --region $Region --stack-name $Stack
aws cloudformation wait stack-delete-complete --region $Region --stack-name $Stack
Write-Host "Stack $Stack deleted."
