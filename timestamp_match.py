import json
import pickle
from datetime import datetime
import utils_lite.configSrc as cfg

cv_activities = [{'class_id': 1, 'action': 'PICK', 'timestamp': '2023-03-15:11:28:53'}, {'class_id': 1, 'action': 'PICK', 'timestamp': '2023-03-15:11:28:53', 'active_zone': 'all_shelves'}, {'class_id': 1, 'action': 'PICK', 'timestamp': '2023-03-15:11:29:50'}]
# with open('archive/f2000ab5-a1ef-4960-bc63-7c34d4ba91f1/ls_activities.pickle', 'rb') as f:
#     ls_activities = pickle.load(f)
ls_activities =  {"user_activity_instance":{"machine_id":"da4d8e2d-c1b2-4145-bf28-8805e6882620","machine_name":"Regus Liberty Station VICKI 1","report_id":"2a2240db-22b9-42aa-8905-de35ec54448e","user_activities":[{"id":"5ac84c04-27f8-4b30-8f68-3fbaa71403ca","user_activity_type":"DOOR_OPENED","row":0,"column":0,"count":0,"in_use_count":0,"previous_count":0,"product_id":-1,"product_name":-1,"activity_time":"2023-03-17:11:24:49","activity_time_str":"2023-03-17:11:24:49"},{"id":"6a1eae65-c608-44dc-9222-38ab74c5cb87","user_activity_type":"USER_PICKUP","row":5,"column":5,"count":3,"in_use_count":1,"previous_count":4,"product_id":"44663bf7-8fcf-41b6-8479-12a6bed5a31b","product_name":"Coca Cola 20z","activity_time":"2023-03-17:11:24:52","activity_time_str":"2023-03-17:11:24:52"},{"id":"30b86511-3eda-4f0c-bcf9-2042a8e1f4ea","user_activity_type":"USER_PICKUP","row":5,"column":4,"count":1,"in_use_count":1,"previous_count":2,"product_id":"44663bf7-8fcf-41b6-8479-12a6bed5a31b","product_name":"Coca Cola 20z","activity_time":"2023-03-17:11:24:54","activity_time_str":"2023-03-17:11:24:54"},{"id":"10e27e00-d8b8-4c94-9ff3-748084bcd590","user_activity_type":"USER_PICKUP","row":5,"column":4,"count":0,"in_use_count":1,"previous_count":1,"product_id":"44663bf7-8fcf-41b6-8479-12a6bed5a31b","product_name":"Coca Cola 20z","activity_time":"2023-03-17:11:24:56","activity_time_str":"2023-03-17:11:24:56"},{"id":"df4a3b82-4dfe-47a0-825f-faedd77ee065","user_activity_type":"DOOR_CLOSED","row":0,"column":0,"count":0,"in_use_count":0,"previous_count":0,"product_id":-1,"product_name":-1,"activity_time":"2023-03-17:11:24:59","activity_time_str":"2023-03-17:11:24:59"}]},"correlation_id":"102.1679077508878177","timestamp":0,"status":0,"error":-1,"message":-1,"path":-1,"Error":-1,"fault":-1}

def adjust_cv_activities_timestamps(cv_activities, ls_activities):
	cv_action_to_ls_action = {'PICK':'USER_PICKUP', 'RETURN':'USER_PUTBACK'}
	user_activities = ls_activities['user_activity_instance']['user_activities']
	for idx_cv, activity in enumerate(cv_activities):
		cv_timestamp = activity['timestamp']
		action = activity['action']
		for idx_ls, user_activity  in enumerate(user_activities):
			#activity type must match
			if user_activity['user_activity_type'] != cv_action_to_ls_action[action] and user_activity['product_name'] != cfg.cls_dict[activity['class_id']]:
				continue
			else:
				ls_timestamp = user_activity['activity_time']
				del user_activities[idx_ls]
				cv_activities[idx_cv]['timestamp'] = ls_timestamp
				print(ls_timestamp)
				break
	return cv_activities

print(cv_activities)
cv_activities = adjust_cv_activities_timestamps(cv_activities, ls_activities)
print('Adjusted----------')
print(cv_activities)