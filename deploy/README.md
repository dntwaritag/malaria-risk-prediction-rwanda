# Deploying to AWS (Lambda container + API Gateway)

**What is deployed:** a small FastAPI web app and JSON API (`app/`) serving the trained logistic model, its model card and
out-of-fold risk scores per survey cluster. **No DHS person-level data and no GPS coordinates are included**, in the image or the repo.

```
Browser / client  ->  API Gateway HTTP API (throttled)  ->  Lambda (container image from ECR, FastAPI via Lambda Web Adapter)
```

## Why this design (note)
* **AWS App Runner is not an option for new accounts.** AWS closed it to new customers on 30 April 2026 and recommends Amazon ECS Express Mode for always-on containers.
* **Lambda is the most cost-efficient fit for a research demo.** You pay per request; with light traffic the cost is close to zero
  (Lambda and API Gateway both have free tiers; the main steady cost is a few cents for ECR storage and CloudWatch logs). Nothing runs, or is billed, while idle.
* **No code changes between laptop and cloud.** The same image runs locally with `docker run -p 8080:8080 malaria-risk`.
* **Throttling built in.** The API allows 10 requests/second (burst 20), which caps cost and abuse on a public URL.
* **Trade-offs:** the first request after a quiet period is slower (cold start, a few seconds while pandas and scikit-learn load);
  uploads are limited to 3 MB (Lambda payload limit); there is no persistent state (none is needed).
* If you later need an always-on service with sub-second responses, move the same image to Amazon ECS Express Mode (it adds an
  Application Load Balancer, which has a steady monthly cost).

## One-time setup
1. Install AWS CLI v2 and Docker Desktop; run `aws configure` (or `aws sso login`) using an IAM identity that can manage CloudFormation, ECR, IAM roles, Lambda, API Gateway and CloudWatch Logs.
2. With the DHS files in `data/raw/` and your `malaria-env` active, create the deploy artefacts:
   `python -m src.train_final`  (writes `models/` and `app/cluster_scores.json`; both are git-ignored, not committed).
3. Run the tests: `python -m pytest -q`.
4. Optional but recommended: create an AWS Budget alert (for example 5 USD/month) in the Billing console.

## Deploy
PowerShell: `.\deploy\deploy.ps1 -Region eu-west-1`      Bash/WSL: `REGION=eu-west-1 ./deploy/deploy.sh`

The script (1) creates the ECR repository, (2) builds a linux/amd64 image and pushes it, (3) creates the Lambda function and API,
then prints the URL and calls `/health`. Re-running deploys a new image tag.
Region note: Africa (Cape Town) `af-south-1` is closest to Rwanda but is an opt-in region that must be enabled in your account first; `eu-west-1` works out of the box.

## Operate
* Logs: CloudWatch Logs group `/aws/lambda/malaria-risk` (kept 30 days).
* Update the model: rerun `python -m src.train_final`, then `deploy.ps1`.
* Remove everything: `.\deploy\destroy.ps1 -Region eu-west-1`.

## Security notes
* HTTPS only; responses carry `nosniff`, `X-Frame-Options` and a strict Content-Security-Policy; uploads are processed in memory and not stored.
* The URL is public and has no login. For a private deployment, add authentication (for example an API Gateway JWT authorizer with Amazon Cognito) before sharing it.
* Lambda runs the container as AWS's sandboxed user with a read-only file system; the IAM role only has permission to write logs.
* Confirm that publishing cluster-level risk scores is allowed under your DHS data-use agreement before sharing the URL widely.
* Not yet verified by me: the image was not built or run end to end here (no Docker daemon in my workspace). The app itself was tested with a clean install of `requirements-app.txt`; the template passes `cfn-lint`. Do a first deploy in a throw-away stack and call `/health`.
