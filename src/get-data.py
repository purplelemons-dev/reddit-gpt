
import requests as r
import yaml

with open("secret.yaml", "r") as f:
    secret = yaml.safe_load(f).get("secret")

print(secret)

#BASE = "https://www.reddit.com/api/v1/access_token"
#
#r.get()
