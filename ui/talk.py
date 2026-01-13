import requests

requests.post(
    "http://localhost:5000/say",
    json={"who": "joe", "text": "Welcome adventures in our tavern. Whant some beer or a story about a dark magician ?"}
)
