import urllib.request
import json
import os

print('=== 1. TEST /api/health ===')
req = urllib.request.urlopen('http://localhost:5000/api/health')
health = json.loads(req.read().decode('utf-8'))
print('Health status:', req.status, health)
assert req.status == 200

print('\n=== 2. TEST /api/config ===')
req = urllib.request.urlopen('http://localhost:5000/api/config')
cfg = json.loads(req.read().decode('utf-8'))
print('Config image_size:', cfg.get('preprocessing', {}).get('image_size'))
print('Config HOG params:', cfg.get('hog'))
print('Config KNN params:', cfg.get('knn'))
assert cfg.get('preprocessing', {}).get('image_size') == [256, 256]

print('\n=== 3. TEST /api/evaluate ===')
req = urllib.request.urlopen('http://localhost:5000/api/evaluate')
eval_data = json.loads(req.read().decode('utf-8'))
print('Evaluate success:', eval_data.get('success'))
print('LOOCV metrics:', eval_data.get('loocv_metrics'))
print('Reference dataset info:', eval_data.get('reference_dataset'))
print('Confusion matrix labels count:', len(eval_data.get('confusion_matrix', {}).get('labels', [])))
print('Per-class count:', len(eval_data.get('per_class_chart', [])))
assert round(eval_data.get('loocv_metrics', {}).get('accuracy'), 2) == 61.94
assert len(eval_data.get('per_class_chart', [])) == 18

print('\n=== 4. REAL HTTP POST /api/verify (NON-HOLDOUT REFERENCE IMAGE) ===')
sample_file = r'D:\Yaevia\backend\dataset\raw\f4a8cfec6aec4502aa6d59aac93502ab.jpeg'
assert os.path.exists(sample_file), 'Sample file not found'

boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
body = bytearray()

# Add ground_truth_name
body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="ground_truth_name"\r\n\r\nIbnu Gayuh Fadilah\r\n'.encode('utf-8'))
# Add ground_truth_nim
body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="ground_truth_nim"\r\n\r\nA710230119\r\n'.encode('utf-8'))
# Add file
body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{os.path.basename(sample_file)}"\r\nContent-Type: image/jpeg\r\n\r\n'.encode('utf-8'))
with open(sample_file, 'rb') as f:
    body.extend(f.read())
body.extend(f'\r\n--{boundary}--\r\n'.encode('utf-8'))

http_req = urllib.request.Request(
    'http://localhost:5000/api/verify',
    data=bytes(body),
    headers={
        'Content-Type': f'multipart/form-data; boundary={boundary}',
        'Content-Length': str(len(body))
    },
    method='POST'
)

resp = urllib.request.urlopen(http_req)
print('HTTP Status:', resp.status)
res = json.loads(resp.read().decode('utf-8'))
print('Response JSON:')
for k, v in res.items():
    if k in ('top_matches', 'k_neighbors_detail'):
        print(f'  {k}: [length {len(v)}]')
    else:
        print(f'  {k}: {v}')

assert res.get('success') == True
assert res.get('feature_vector_length') == 34596
assert res.get('predicted_name') == 'Ibnu Gayuh Fadilah'
assert res.get('is_correct') == 1

print('\n>>> ALL LIVE HTTP API TESTS PASSED! <<<')
