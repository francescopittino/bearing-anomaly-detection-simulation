import random
import time
import threading
import logging
import json
from queue import Queue, Empty
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from paho.mqtt import client as mqtt_client

import tensorflow as tf
from keras.layers import Input, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers

"""
BearingEvaluator Class file

offers the reception and evaluation part of the simulation, where a thread receives the data and
places it in two queues, one for training and one for operational usage.

After the training queue gets to the required amount, a model is trained and run on any
later values, the error is saved and can be plotted at the end.
It will issue a warning when 10 consecutive anomalies (values above a threshold calculated after training) appear.

Publishes on [bearing]/results topic
"""

#create a logger
anomaly_logger = logging.getLogger("AnomalyLogger")
anomaly_logger.setLevel(logging.WARNING)
if not anomaly_logger.handlers:
    fh = logging.FileHandler("ANOMALY.log")
    fh.setFormatter(logging.Formatter('%(asctime)s-%(message)s'))
    anomaly_logger.addHandler(fh)

class BearingEvaluator:
    def __init__(self, broker: str, port: int, topic: str, train_amount: int = 1500, epochs: int = 100, batch_size: int = 10):
        self.broker = broker
        self.port = port
        self.topic = topic
        self.train_amount = train_amount
        self.epochs = epochs
        self.batch_size = batch_size
        #create random client id
        self.client_id = f'subscribe-{random.randint(0, 1000)}'
        #create training value and operational value queues
        self.train_queue = Queue()
        self.op_queue = Queue()
        
        self.is_transmitting = True
        self._stop_event = threading.Event()
        
        self.streaming_results = None
        self.trigger_10_idx = None
        #temp value, will be decided based on quantile
        self.auto_threshold = 0.0

        self._mqtt_thread = None
        self._monitor_thread = None
        self.client = None
        
    #connection function
    def _connect_mqtt(self) -> mqtt_client:
        def on_connect(client, userdata, flags, reason_code, properties):
            if reason_code == 0:
                print(f"[{self.topic} Evaluator] Connected to MQTT Broker!")
            else:
                print(f"[{self.topic} Evaluator] Connection failed (Code: {reason_code})")

        client = mqtt_client.Client(
            client_id=self.client_id,
            callback_api_version=mqtt_client.CallbackAPIVersion.VERSION2,
        )
        client.on_connect = on_connect
        client.connect(self.broker, self.port) #decided from the main file
        return client

    #on message reception function: put value in the right queue or stop operations
    def _on_message(self, client, userdata, msg):
        try:
            #parse json
            payload_str = msg.payload.decode()
            data = json.loads(payload_str)
            
            val = data.get("value")
            file_name = data.get("file")
            
            if file_name == "STOP":
                self.is_transmitting = False
                print(f"[{self.topic} Evaluator] received STOP.")
                return #end here

            #build df using value and file (used for raw buffer, NOT FULLY IMPLEMENTED YET)
            new_value = pd.DataFrame({f"{msg.topic}": float(val)}, index=[pd.Timestamp.now()]) #timestamp is now, not that of the NASA dataset
            new_value.attrs["file"] = file_name

            if self.train_queue.qsize() <= self.train_amount:
                self.train_queue.put(new_value)
            else:
                self.op_queue.put(new_value)
                
        except json.JSONDecodeError:
            #TEMP values as plaintext
            payload = msg.payload.decode()
            if payload == "STOP":
                self.is_transmitting = False
                print(f"[{self.topic} Evaluator] received STOP.")
            else: #if not stop, then it's a value
                try:
                    new_value = pd.DataFrame({f"{msg.topic}": float(payload)}, index=[pd.Timestamp.now()])
                    if self.train_queue.qsize() <= self.train_amount:
                        self.train_queue.put(new_value)
                    else:
                        self.op_queue.put(new_value)
                except ValueError:
                    pass

    def _mqtt_loop(self): #main loop for the subscriber
        self.client = self._connect_mqtt()
        self.client.subscribe(self.topic)
        self.client.on_message = self._on_message
        self.client.loop_forever()

    #function to create a LSTM Autoencoder, design by github.com/BLarzalere
    def _create_model(self, X): 
        inputs = Input(shape=(X.shape[1], 1))
        L1 = LSTM(16, activation='relu', return_sequences=True, kernel_regularizer=regularizers.l2(0.00))(inputs)
        L2 = LSTM(4, activation='relu', return_sequences=False)(L1)
        L3 = RepeatVector(X.shape[1])(L2)
        L4 = LSTM(4, activation='relu', return_sequences=True)(L3)
        L5 = LSTM(16, activation='relu', return_sequences=True)(L4)
        output = TimeDistributed(Dense(1))(L5)
        return Model(inputs=inputs, outputs=output)

    #main loop for the monitor
    def _monitor_loop(self):
        #wait for the training values to be ready
        while self.train_queue.qsize() < self.train_amount and not self._stop_event.is_set():
            time.sleep(0.1)

        if self._stop_event.is_set():
            return
        #build training set
        training_set = pd.DataFrame()
        while not self.train_queue.empty():
            training_set = pd.concat([training_set, self.train_queue.get()])
            
        #sort and normalize 
        training_set.sort_index(inplace=True)
        scaler = preprocessing.MinMaxScaler()
        norm_train = scaler.fit_transform(training_set.values)
        train = pd.DataFrame(norm_train.reshape(norm_train.shape[0], 1), index=training_set.index)

        #create and train model
        model = self._create_model(train)
        model.compile(optimizer='adam', loss='mse')
        print(f"[{self.topic} Evaluator] training autoencoder model...")
        model.fit(train, train, epochs=self.epochs, batch_size=self.batch_size, validation_split=0.05, verbose=0)
        print(f"[{self.topic} Evaluator] model trained successfully.")

        #calculate predictions on training values
        train_reshaped = train.values.reshape(train.shape[0], 1, 1)
        train_predictions = model.predict(train_reshaped, verbose=0)

        train_actuals_flat = train.values[:, 0]
        train_preds_flat = train_predictions[:, 0, 0]
        train_reconstruction_error = pd.Series((train_actuals_flat - train_preds_flat) ** 2)
        
        #set the 0.9999 quantile as threshold
        self.auto_threshold = train_reconstruction_error.quantile(0.9999).item()
        print(f"[{self.topic} Evaluator] Threshold set to: {self.auto_threshold:.6f}")

        streamed_timestamps = []
        streamed_errors = []
        consecutive_anomalies = 0
        
        #main loop, accepts values acquired by the subscriber thread from the queue and runs the model on those
        while (self.is_transmitting or not self.op_queue.empty()) and not self._stop_event.is_set():
            try:
                #attempt to get the value without waiting
                df = self.op_queue.get_nowait()
                actual_value = float(df.iloc[-1, 0])
                scaled_value = scaler.transform([[actual_value]])[0][0]
                current_input = np.array([[[scaled_value]]])

                #runs the model on the scaled value
                pred = model.predict(current_input, verbose=0)
                pred_value = pred[0, 0, 0]

                #calculate error
                error = (scaled_value - pred_value) ** 2
                
                #update consecutive anomalies count
                if error > self.auto_threshold:
                    consecutive_anomalies += 1
                else:
                    consecutive_anomalies = 0

                is_triggered = False
                #if you have 10 consecutive anomalies, issue a warning
                if consecutive_anomalies == 10 and self.trigger_10_idx is None:
                    self.trigger_10_idx = df.index[0]
                    is_triggered = True
                    #print it and log it
                    alert_msg = f"[{self.topic}] TRIGGER EXCEEDED: 10 Consecutive anomalies at {self.trigger_10_idx}"
                    print(alert_msg)
                    anomaly_logger.warning(alert_msg)
                
                #create another payload and 
                live_payload = json.dumps({
                    "error": error, 
                    "threshold": self.auto_threshold,
                    "triggered": is_triggered,
                    "file": df.attrs.get("file", "unknown")
                })#publish it to mqtt
                self.client.publish(f"{self.topic}/results", live_payload)
                #save streaming data timestampts and calculated error values
                streamed_timestamps.append(df.index[0])
                streamed_errors.append(error)
            except Empty:
                time.sleep(0.01)

        self.streaming_results = pd.DataFrame({'Reconstruction_Error': streamed_errors}, index=streamed_timestamps)

    def start(self):
        #starts the threads
        self._stop_event.clear()
        self._mqtt_thread = threading.Thread(target=self._mqtt_loop, daemon=True)
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._mqtt_thread.start()
        self._monitor_thread.start()

    def stop(self):#stop event = disconnect
        self._stop_event.set()
        if self.client:
            self.client.disconnect()
            
    #join helper method
    def join(self, timeout=None):
        if self._monitor_thread: 
            self._monitor_thread.join(timeout=timeout)

    #function to plot the results acquired during the simulation, marking the point where the 10 consecutive anomalies first issued a warning
    def plot_results(self):
        if self.streaming_results is None or self.streaming_results.empty:
            print(f"[{self.topic} Evaluator] No data available for plotting.")
            return

        fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
        ax.plot(self.streaming_results.index, self.streaming_results['Reconstruction_Error'], color='purple', label='Reconstruction Error (MSE)', linewidth=1.5)
        ax.axhline(y=self.auto_threshold, color='red', linestyle='-', label='Calculated Threshold')

        if self.trigger_10_idx is not None:
            ax.axvline(x=self.trigger_10_idx, color='orange', linestyle='--', linewidth=2, label='First 10 Consecutive Anomalies')

        ax.set_yscale('log') #use logaritmic scale
        ax.set_title(f'Streaming Reconstruction Error [{self.topic}]', fontsize=16)
        ax.set_ylabel('Reconstruction Error (Log MSE)')
        ax.set_xlabel('Time')
        ax.grid(True, which="both", linestyle="--", alpha=0.6)
        plt.xticks(rotation=45)
        ax.legend(loc='upper left')
        plt.tight_layout()
        plt.show()