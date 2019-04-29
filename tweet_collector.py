import subprocess
import csv

house_handles = "house_handles.txt"
senate_handles = "senator_handles.txt"

def load_congress_handles():
    handles = []
    with open(house_handles) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            handles += row

    with open(senate_handles) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            handles += row
    
    return handles

def scrape_tweets(congress_member_handle):
    
    command = "/usr/local/bin/twitterscraper"
    filename = "Tweets/" + congress_member_handle + ".json"
    # /usr/local/bin/twitterscraper -u amyklobuchar --begindate 2018-01-01 --output=tweets.json
    subprocess.call([command, "-u", congress_member_handle, "--begindate",
        "2017-01-01", "--output", filename])

def main():
    handles = load_congress_handles()
    for handle in handles:
        print("Starting " + handle)
        scrape_tweets(handle)



main()
