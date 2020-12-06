from kafka  import KafkaProducer
import random
import logging
from time import sleep
from json import dumps
import pandas as pd


def encode_as_json(movement_train):
	x = dumps(movement_train.loc[:, movement_train.columns != 'locked'].tolist())
	y = dumps(movement_train[:,-1:].tolist())
	return [x, y]

def generate_stream(**kwargs):
	# We create a Kafka producer
	producer = KafkaProducer(bootstrap_servers=['kafka:9092'], value_serializer=lambda x: dumps(x).encode('utf-8'))

	# Create some sample data, for demonstration purposes, we just take som samples from the initial training data
	# This could be your continous flow of incoming data
	movements_stream_input = pd.read_csv("../../../data/SmartMovementExport.csv")
	# From the whole input set, take random index for 500 new training examples
	rand = random.sample(range(0, len(movements_stream_input)), 500)
	logging.info('We stream now over Kafka some data.', producer.partitions_for('MovementsTopic'))

	for i in rand:
		json_stream_data = encode_as_json(movements_stream_input[i])
		producer.send('MovementsTopic', value=json_stream_data)
		sleep(1)

	producer.close()
