import os
import random
import time
import json
import threading
from tqdm import tqdm
import pandas as pd
import numpy as np
from paho.mqtt import client as mqtt_client

"""
Bearing publisher Class file

offers the streaming part of the simulation.

It simulates a sensor by calculating the RMS for every file (for the third test, every file is a 1-second snapshot, recorded every 10-minutes)
and publishing it on a mqtt topic.

It is interpreted as the sensor publishing the RMS for every interval to limit the transmission, i.e. the rms for the last 30 seconds.
When an anomaly arises, the raw buffer can be used for a wavelet analysis or something else (TO BE IMPLEMENTED)

"""

class BearingPublisher:
    def __init__(self, broker: str, port: int, topic: str, folder: str, column: int = 0, pause: float = 0.05):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.folder = folder
        self.column = column
        self.pause = pause
        #random client id
        self.client_id = f'publish-{random.randint(0, 1000)}'
        self.client = None
        self._stop_event = threading.Event()
        self._thread = None
        
        #raw buffer
        self.current_raw_buffer = []

    #mqtt connection function
    def _connect_mqtt(self) -> mqtt_client:
        def on_connect(client, userdata, flags, reason_code, properties):
            if reason_code == 0:
                print(f"[{self.topic} Publisher] connected, listening for commands")
                client.subscribe(f"{self.topic}/command") #topic where commands can be issued

        client = mqtt_client.Client(
            client_id=self.client_id,
            callback_api_version=mqtt_client.CallbackAPIVersion.VERSION2,
        )
        client.on_connect = on_connect
        client.connect(self.broker, self.port)
        return client

    def _on_message(self, client, userdata, msg):
        try:
            command = msg.payload.decode()
            if command == "REQUEST_BUFFER": #there has been a command requesting a buffer
                print(f"[{self.topic} Publisher] buffer request received")
                #send buffer on topic
                payload = json.dumps({"buffer": self.current_raw_buffer})
                client.publish(f"{self.topic}/buffer", payload)
        except Exception as e:
            print(f"[{self.topic} Publisher ERROR]: {e}")

    #main publisher loop
    def _publish_loop(self):
        self.client = self._connect_mqtt()
        self.client.on_message = self._on_message
        self.client.loop_start()

        files = sorted(os.listdir(self.folder))
        #for every test file
        for file in tqdm(files, desc=f"Publishing {self.topic}"):
            if self._stop_event.is_set():
                break

            file_path = os.path.join(self.folder, file)
            #get the file as dataframe
            df = pd.read_csv(file_path, sep='\t')
            
            #save the values for this bearing to the raw buffer
            self.current_raw_buffer = df.iloc[:, self.column].tolist()

            #calculate rms
            df_rms = np.array(np.sqrt(np.mean(df**2, axis=0)))
            df_rms = pd.DataFrame(df_rms.reshape(1, 4))
            value = float(df_rms.iloc[0, self.column])
            
            #stream it to mqtt topic
            payload = json.dumps({"value": value, "file": file})
            self.client.publish(self.topic, payload)
            
            time.sleep(self.pause) #pause for the given amount of time

        self.client.publish(self.topic, json.dumps({"value": 0.0, "file": "STOP"})) #end value to stop the evaluators
        time.sleep(0.5)
        self.client.loop_stop()
        self.client.disconnect()

    def start(self):
        self._stop_event.clear()
        #start main publisher loop
        self._thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        
    #join helper function
    def join(self, timeout=None):
        if self._thread:
            self._thread.join(timeout=timeout)