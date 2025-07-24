import boto3
import json

#model_id
model_id = "amazon.nova-micro-v1:0"

#prompt
message_list = [{"role":"user", "content": [{"text":"One liner for a hello world program in python"}]}]

#system prompt
system_list = [{"text": "Act as a creative writing assistant. When the user provides you with a short story about that topic"}]

#model parameters
model_params = {
    "temperature": 0.7,
    "topP": 0.9,
    "maxTokens": 500,
    "topK": 20
}

#invoke model
bedrock = boto3.client(service_name='bedrock-runtime')

body = json.dumps({
    "schemaVersion": "messages-v1",
    "messages": message_list,
    "system": system_list,
    "inferenceConfig": model_params
})

#invoke model
response = bedrock.invoke_model(modelId=model_id, body=body)
print(response)
#decode response body
model_response = json.loads(response["body"].read())
print("\n ==================================================== ")
print(model_response)


"""
{'ResponseMetadata': {'RequestId': '9b11a83a-1f82-4185-a544-ea99ec26a57c', 'HTTPStatusCode': 200, 
'HTTPHeaders': {'date': 'Thu, 24 Jul 2025 17:33:04 GMT', 'content-type': 'application/json', 'content-length': '418', 'connection': 'keep-alive', 
'x-amzn-requestid': '9b11a83a-1f82-4185-a544-ea99ec26a57c', 'x-amzn-bedrock-invocation-latency': '244', 'x-amzn-bedrock-cache-write-input-token-count': '0', 
'x-amzn-bedrock-cache-read-input-token-count': '0', 'x-amzn-bedrock-output-token-count': '51', 'x-amzn-bedrock-input-token-count': '28'}, 'RetryAttempts': 0}, 
'contentType': 'application/json', 'body': <botocore.response.StreamingBody object at 0x7e3686ca38b0>}

 ==================================================== 
{'output': {'message': {'content': [{'text': 'Sure, here\'s a one-liner for a "Hello, World!" program in Python:\n\n```python\nprint("Hello, World!")\n```\n\nThis simple line of code prints the message "Hello, World!" to the console when executed.'}], 'role': 'assistant'}}, 'stopReason': 'end_turn', 'usage': {'inputTokens': 28, 'outputTokens': 51, 'totalTokens': 79, 'cacheReadInputTokenCount': 0, 'cacheWriteInputTokenCount': 0}}
"""