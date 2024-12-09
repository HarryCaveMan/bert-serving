from locust import HttpUser, task, between
from json import load as from_json_file
from random import randint

with open("scenarios.json","r") as testfile:
    scenarios = from_json_file(testfile)

class LoadTester(HttpUser):
    wait_time = between(0.2, 0.8)

    def on_start(self):
        i = randint(0,len(scenarios)-1)
        self.scenario = scenarios[i]

    @task
    def encode(self):
        self.client.post("/encode",json=self.scenario)