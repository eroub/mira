# Get the date in a flexible format
# Matches 2023-2-6 and 2023-02-06
today=$(date +%Y-%-m-%-d)

# Get the date two days ago in a flexible format
two_days_ago=$(date -v -2d +%Y-%-m-%-d)

# Construct the file name for the two days ago file
file_name="BTCUSDT-trades-${two_days_ago}.parquet"

# Check if the file exists and delete it
if [ -f "$file_name" ]; then
  rm "$file_name"
fi
