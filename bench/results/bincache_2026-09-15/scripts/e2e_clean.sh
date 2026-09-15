#!/bin/sh
# delete the e2e partition and any inbox left by it (test objects only)
( . "$HOME/.mojolearn_r2"; export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" AWS_DEFAULT_REGION=auto
  EP="https://$R2_ACCOUNT_ID.r2.cloudflarestorage.com"
  aws s3 rm "s3://$R2_BUCKET/bincache/v1/none/local-e2e-selftest/" --recursive --endpoint-url "$EP" --only-show-errors
  aws s3 rm "s3://$R2_BUCKET/bincache/inbox/" --recursive --endpoint-url "$EP" --only-show-errors
  echo "R2 objects under bincache/ after cleanup: $(aws s3 ls "s3://$R2_BUCKET/bincache/" --recursive --endpoint-url "$EP" | wc -l | tr -d ' ')" )
