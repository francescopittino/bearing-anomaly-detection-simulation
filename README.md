# bearing-anomaly-detection
 
This repository simulates an anomaly detection pipeline using the NASA Bearing dataset (https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset), using MQTT for communication and LSTM Autoencoders for anomaly detection.

# EN/IT

# EN
# What does it do:
- Simulates the streaming of data from a sensor by calculating the Root Mean Square for every test file (1-second snapshots every ten minutes) and publishing it to an MQTT Topic.
- Simulates the detection of anomalies using an autoencoder model that is trained on the initial values from the sensor and is then used to classify anomalies based on the model's loss value.
- If the consecutive values exceeding a certain threshold cross a predefined amount, it issues a warning on an MQTT Topic.
- At the end of the simulation, a plot is shown, allowing you to visualize the error values and how the bearing health deteriorates.

# The future plan:
- Using the already implemented raw buffer request system and the warning issued on an MQTT Topic, I plan to implement another system that will listen for warning and, when one is received, request the raw buffer for that bearing. With this raw buffer, a wavelet analysis (and possibly something else) will be performed to identify if the bearing is faulty (and if so try to determine what kind of fault it is) or if the spike in loss is produced by cross-talk from another bearing.
- A GUI is not a possibility, but not a priority.

There is a description at the beginning of every file that explains what it is used for in more detail.


# IT
# Cosa fa:
- Simula lo streaming di dati da un sensore, calcolando la RMS per ogni file di test (lunghi 1 secondo di misurazioni ogni 10 minuti) e pubblicandolo sul topic MQTT.
- Simula il riconoscimento di anomalie usando un autoencoder addestrato sui valori iniziali trasmessi, poi usato per classificare le anomalie in base al valore di perdita.
- Se dei valori superano la soglia calcolata per un ammontare di valori consecutivi predefinito, trasmette una allerta su un topic MQTT.
- Alla fine della simulazione, un grafico mostrato permette di visualizzare i valori di errore durante la simulazione, indicatori di come la salute del cuscinetto è degenerata nel tempo.

# I piani futuri:
- Utilizzando il già implementato sistema di richiesta del buffer (su cui è calcolata la RMS) e il sistema di allerta tramite MQTT, implementare un sistema che rimanga in ascolto per allerte e, alla ricezione, richieda il buffer del cuscinetto corrispondente e lo utilizzi per una analisi più approfondita (es: wavelet) per determinare la causa dei valori elevati, come un guasto (e che tipo di guasto) o rumore trasmesso dagli altri cuscinetti.
- Una interfaccia grafica potrebbe essere implementata, ma non è una priorità.

Una descrizione iniziale in ogni file descrive in maniera più approfondita il loro funzionamento.

# End-Of-Test Images (Plots):
<img width="1120" height="480" alt="sim_bearing_1_loss" src="https://github.com/user-attachments/assets/171a9754-91fd-4180-afde-412add8e566d" />
<img width="1120" height="480" alt="sim_bearing_2_loss" src="https://github.com/user-attachments/assets/f411b703-1b54-4aa5-8262-cf2a453d0928" />
<img width="1120" height="480" alt="sim_bearing_3_loss" src="https://github.com/user-attachments/assets/cb8a9b4d-0ef3-43a9-a307-348e93929dc9" />
<img width="1120" height="480" alt="sim_bearing_4_loss" src="https://github.com/user-attachments/assets/a5822820-fa2f-4362-a1a6-8d2b8057dd48" />
