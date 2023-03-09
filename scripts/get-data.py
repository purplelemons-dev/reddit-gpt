
import praw
from yaml import safe_load

with open("./src/resources/config.yaml", "r") as f:
    config = safe_load(f)


reddit = praw.Reddit(
    client_id=config["client-id"],
    client_secret=config["secret"],
    user_agent="reddit-gpt-bot/0.0.1"
)

test=reddit.submission(id="11diphg")

print(f"{test.title=}")
print(f"{test.selftext=}")
for comment in test.comments[0:10]:
    print(f"{comment.author=}")
    print(f"{comment.ups=}")
    print(f"{comment.body=}")

#for submission in reddit.subreddit("all").top(limit=10):
#    print(f"{submission.title=}")
#    print(f"{submission.comments=}")
