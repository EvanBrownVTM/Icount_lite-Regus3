#!/usr/bin/env python
import pika
import json

credentials = pika.PlainCredentials('nano','nano')
parameters = pika.ConnectionParameters('localhost',5672,'/',credentials)
connection = pika.BlockingConnection(parameters)
channel = connection.channel()

channel.queue_declare(queue="cvPost",durable = True)
#data = "{'cmd': 'Done', 'transid': '683d9025-fb72-4688-93a6-9fb3fbcaec37', 'timestamp': '20230301-16_52_14', 'cv_activities': [], 'ls_activities': '{"user_activity_instance":{"machine_id":"da4d8e2d-c1b2-4145-bf28-8805e6882620","machine_name":"Regus Liberty Station VICKI 1","report_id":"5802d23e-e7a7-49e6-aa19-7442e666c953","user_activities":[{"id":"73fa86f4-3454-42ed-87e2-fa34e505fa4d","user_activity_type":"DOOR_OPENED","row":0,"column":0,"count":0,"in_use_count":0,"previous_count":0,"product_id":null,"product_name":null,"activity_time":"2023-03-01:16:51:57","activity_time_str":"2023-03-01:16:51:57"},{"id":"d9365587-88af-432c-96c1-dd72e11870e6","user_activity_type":"USER_PICKUP","row":2,"column":7,"count":5,"in_use_count":1,"previous_count":6,"product_id":"eec36681-4cad-45d3-9ef7-c2c598a6d07a","product_name":"Red Bull Blue 12oz","activity_time":"2023-03-01:16:52:00","activity_time_str":"2023-03-01:16:52:00"},{"id":"ec436cad-2b7c-4e98-960f-62885fc5ca04","user_activity_type":"DOOR_CLOSED","row":0,"column":0,"count":0,"in_use_count":0,"previous_count":0,"product_id":null,"product_name":null,"activity_time":"2023-03-01:16:52:04","activity_time_str":"2023-03-01:16:52:04"}]}}"
data = '{\n "cmd": "Done", \n "transid":"Testtrans"\n, "cv_activities":[]\n, "ls_activities":[]\n}'

#data = '{\n "src": "all", \n "parm1":"trans2"\n}'
mess = json.dumps(data)
mess =json.loads(mess)

channel.basic_publish(exchange='',
                        routing_key="cvPost",
                        body=mess)

print(" [x] Sent data %", data)
connection.close()
