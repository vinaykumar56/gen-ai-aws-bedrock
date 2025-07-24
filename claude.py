import boto3
import json

#model_id
model_id = "anthropic.claude-3-haiku-20240307-v1:0"

#prompt
prompt = "One liner for a hello world program in python"

#system prompt
# system_list = [{"text": "Act as a creative writing assistant. When the user provides you with a short story about that topic"}]

# #model parameters
# model_params = {
#     "temperature": 0.7,
#     "topP": 0.9,
#     "maxTokens": 500,
#     "topK": 20
# }

#invoke model
bedrock = boto3.client(service_name='bedrock-runtime')

body = json.dumps({
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 512,
    "temperature": 0.5,
    "messages": [{
        "role": "user",
        "content": [{"type":"text", "text":prompt}]
    }]
})

#invoke model
response = bedrock.invoke_model(modelId=model_id, body=body)
print(response)
#decode response body
model_response = json.loads(response["body"].read())
print("\n ==================================================== ")
print(model_response)


"""
{'ResponseMetadata': {'RequestId': 'd1d4b6e5-45b3-4b5e-823c-ea5f117e9374', 'HTTPStatusCode': 200, 'HTTPHeaders': {'date': 'Thu, 24 Jul 2025 17:57:22 GMT', 'content-type': 'application/json', 'content-length': '266', 'connection': 'keep-alive', 'x-amzn-requestid': 'd1d4b6e5-45b3-4b5e-823c-ea5f117e9374', 'x-amzn-bedrock-invocation-latency': '228', 'x-amzn-bedrock-output-token-count': '9', 'x-amzn-bedrock-input-token-count': '16'}, 'RetryAttempts': 0}, 'contentType': 'application/json', 'body': <botocore.response.StreamingBody object at 0x729b97953100>}
"""

"""
{'id': 'msg_bdrk_01AFSvnMq86Ne5XyJwXKFdnL', 'type': 'message', 'role': 'assistant', 'model': 'claude-3-haiku-20240307', 'content': [{'type': 'text', 'text': 'print("Hello, World!")'}], 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 16, 'output_tokens': 9}}
"""