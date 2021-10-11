import requests
import json

# client_id 为官网获取的AK， client_secret 为官网获取的SK
host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=DXN8o5eNaheZahxK558I8GOs&client_secret=bBYbDZj6dbv7w5Pr5hqelm7lL8GfPsBR'
response = requests.get(host)
if response:
    # print(response.json()['access_token'])
    access_token = response.json()['access_token']
'''
人脸对比
'''

request_url = "https://aip.baidubce.com/rest/2.0/face/v3/match"

params = "[{\"image\": \"http://xinan.ziqiang.net.cn/detect_face_1.jpg\", \"image_type\": \"BASE64\", \"face_type\": \"LIVE\", \"quality_control\": \"LOW\"},{\"image\": \"http://xinan.ziqiang.net.cn/detect_face_3.jpg\", \"image_type\": \"BASE64\", \"face_type\": \"IDCARD\", \"quality_control\": \"LOW\"}]"
params = json.loads(params)
request_url = request_url + "?access_token=" + access_token
headers = {'content-type': 'application/json'}
response = requests.post(request_url, json=params, headers=headers)
if response:
    print (response.json())