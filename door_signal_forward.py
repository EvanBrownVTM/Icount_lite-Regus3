import time
import utils_lite.configSrc as cfg
import logging
import pika
import sys
import json

#RabbitMQ Initialization
def initializeChannels(logger, queue_list, ip, credentials = pika.PlainCredentials('nano','nano')):
	'''
		queue_list = ['cvRequest', 'cvPost']
		ip = 'localhost'
	'''
	logger.info('Initializing RMQ queues: ' + ' '.join(queue_list) + ' at : ' + ip)
	credentials = pika.PlainCredentials('nano','nano')
	parameters = pika.ConnectionParameters(ip, 5672, '/', credentials, heartbeat=0, blocked_connection_timeout=3000)
	connection = pika.BlockingConnection(parameters)
	channel_list = []
	for queue in queue_list:
		channel = connection.channel()
		channel.queue_declare(queue=queue, durable=True)
		channel.queue_purge(queue=queue)
		channel_list.append(channel)
	logger.info('   Success- queues initialized')
	return channel_list, connection

def main():
	reconnect_interval = 60 #seconds to wait before attempting to reinitialize demographics channel

	#initialize logging
	logging.getLogger("pika").setLevel(logging.WARNING)
	logging.getLogger('requests').setLevel(logging.WARNING)
	logging.getLogger("tensorflow").setLevel(logging.ERROR)
	logging.basicConfig(filename='{}logs/Icount.log'.format(cfg.log_path), level=logging.DEBUG, format="%(asctime)-8s %(levelname)-8s %(message)s")
	logging.disable(logging.DEBUG)
	logger=logging.getLogger()
	logger.info("")
	sys.stderr.write=logger.error

	#initialize channel to receive door signal
	channels, connection = initializeChannels(logger, ['cvRequest'], 'localhost')
	(channel_receive,) = channels

	#initialize channel to forward door signal to icount (local/xavier)
	channels, connection = initializeChannels(logger, ['cvIcount'], 'localhost')
	(channel_icount,) = channels

	#initialize channel to forward door signal to demographics (nano)
	channel_demographics = None
	start_time = time.time()
	if cfg.FaceRec:
		try:
			channels, connection = initializeChannels(logger, ['cvFace'], cfg.IP_ADDRESS_NANO)
			(channel_demographics,) = channels
		except Exception as e:
			logger.info('ERROR: failed to init RMQ channel /door signal to Nano at: ' + cfg.IP_ADDRESS_NANO)
			logger.info('   ' + str(e))

	#receive and forward signals
	while True:
		#receive door signal
		_,_,recv = channel_receive.basic_get('cvRequest')
		if recv:
			logger.info(str(recv, 'utf-8'))
			#forward message to icount
			channel_icount.basic_publish(exchange='',
					routing_key="cvIcount",
					body=recv
					)

			#forward message to demographics, if channel exists
			if channel_demographics is not None:
				channel_demographics.basic_publish( exchange='',
					routing_key="cvFace",
					body=recv
					)
			elif (time.time() - start_time) % reconnect_interval == 0 and cfg.FaceRec:
				try:
					channels, connection = initializeChannels(logger, ['cvFace'], cfg.IP_ADDRESS_NANO)
					(channel_demographics,) = channels
				except Exception as e:
					logger.info('ERROR: failed to init RMQ channel / door signal to Nano at: ' + cfg.IP_ADDRESS_NANO)
					logger.info('   ' + str(e))

if __name__ == '__main__':
	main()
