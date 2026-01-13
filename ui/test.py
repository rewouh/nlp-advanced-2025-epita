import requests

requests.post(
    "http://localhost:5000/scene",
    json={"scene": "guard_barracks"}
)
