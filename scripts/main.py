import time
from bearing_publisher import BearingPublisher
from bearing_evaluator import BearingEvaluator

"""
Main file used to run the simulation on the third test.
Starts the evaluators (receivers+models) and publishers(simulating the bearing sensors)
When the data stream is over, it displays the plots for each bearing, showing their model's loss during the simulation
and highlights the moment the threshold was crossed (10 consecutive anomalies).

ADJUST:
BROKER_IP -> your broker's IP
BROKER_PORT -> your broker's port
DATA_FOLDER -> the location of the test files (i.e.: path to the 3rd test files)
"""

BROKER_IP = "192.168.1.116"
BROKER_PORT = 1883
DATA_FOLDER = r"E:\Repos\bearing-anomaly-detection\bearing-dataset\versions\1\3rd_test\4th_test\txt"

def main():
    
    production_line = ["Bearing 1", "Bearing 2", "Bearing 3", "Bearing 4"]
    evaluators = []
    publishers = []

    print("Initializing evaluators and publishers")
    
    #start evaluators and publishers for the four bearings
    for index, bearing_topic in enumerate(production_line):
        evaluator = BearingEvaluator(
            broker=BROKER_IP, port=BROKER_PORT, topic=bearing_topic,
            train_amount=2500, epochs=100, batch_size=10
        )#models will be training on the first 2500 values, 100 epochs
        evaluators.append(evaluator)

        publisher = BearingPublisher(
            broker=BROKER_IP, port=BROKER_PORT, topic=bearing_topic,
            folder=DATA_FOLDER, column=index, pause=0.05
        )#publishers will pause for 0.05 between values
        publishers.append(publisher)

    #start evaluators, then publishers
    print("Starting evaluators and publishers")
    for evaluator in evaluators:
        evaluator.start()
    for publisher in publishers:
        publisher.start()

    try:
        #while the publishers are still transmitting values
        while any(p._thread.is_alive() for p in publishers):
            time.sleep(0.1) #do nothing and sleep
            
        #wait for publishers and evaluators to finish
        #evaluators will receive a stop message
        for publisher in publishers:
            publisher.join()
        for evaluator in evaluators:
            evaluator.join()
            
        print("Pipeline execution completed for all bearings.")
        for evaluator in evaluators:
            evaluator.plot_results()
        
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt received")
        for publisher in publishers:
            publisher.stop()
        for evaluator in evaluators:
            evaluator.stop()

    input("Press ENTER to close all windows and exit ->") #close the program

if __name__ == '__main__':
    main()