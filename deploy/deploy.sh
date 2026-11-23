#!/usr/bin/env bash
# Deploy the malaria risk service to AWS (Lambda container + API Gateway) (macOS / Linux / WSL). Run from the repository root:
#   REGION=eu-west-1 ./deploy/deploy.sh
set -euo pipefail
REGION="${REGION:-eu-west-1}"; STACK="${STACK:-malaria-risk}"; SERVICE="${SERVICE:-malaria-risk}"
command -v aws >/dev/null && command -v docker >/dev/null || { echo "aws CLI and docker are required"; exit 1; }
for p in models/malaria_risk_model.joblib models/model_card.json app/cluster_scores.json; do
  [ -f "$p" ] || { echo "Missing $p. Run: python -m src.train_final"; exit 1; }
done
ACCT=$(aws sts get-caller-identity --query Account --output text --region "$REGION")
TAG=$(date +%Y%m%d-%H%M%S); REGISTRY="$ACCT.dkr.ecr.$REGION.amazonaws.com"
echo "AWS account $ACCT, region $REGION"
aws cloudformation deploy --region "$REGION" --stack-name "$STACK" --template-file deploy/cloudformation.yaml \
  --capabilities CAPABILITY_IAM --parameter-overrides ServiceName="$SERVICE" CreateService=false --no-fail-on-empty-changeset
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"
docker build --platform linux/amd64 --provenance=false -t "$REGISTRY/$SERVICE:$TAG" .
docker push "$REGISTRY/$SERVICE:$TAG"
aws cloudformation deploy --region "$REGION" --stack-name "$STACK" --template-file deploy/cloudformation.yaml \
  --capabilities CAPABILITY_IAM --parameter-overrides ServiceName="$SERVICE" CreateService=true ImageTag="$TAG" --no-fail-on-empty-changeset
URL=$(aws cloudformation describe-stacks --region "$REGION" --stack-name "$STACK" --query "Stacks[0].Outputs[?OutputKey=='ServiceUrl'].OutputValue" --output text)
echo "Service URL: $URL"; curl -fsS "$URL/health"; echo
