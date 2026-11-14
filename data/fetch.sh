#!/usr/bin/env bash
# Download raw SNAP QC CSV zips into data/raw/ and unzip. URLs are the public download links listed in data/README.md.
# State forms (data/raw/forms/) were downloaded manually and are not fetched here.
set -euo pipefail
cd "$(dirname "$0")/raw"
UA="Mozilla/5.0"   # the site needed a browser user agent
get() { # year, path
  mkdir -p "fy$1"
  curl -fL -A "$UA" -o "fy$1/qcfy$1_csv.zip" "https://snapqcdata.net/sites/default/files/$2/qcfy$1_csv.zip"
  unzip -o -d "fy$1" "fy$1/qcfy$1_csv.zip"
}
get 2022 2024-05
get 2023 2025-03
get 2024 2026-08
