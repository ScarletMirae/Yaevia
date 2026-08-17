"""Quick test verifikasi endpoint."""
import urllib.request, json, cv2, numpy as np

# Buat gambar test
img = np.ones((128, 128), dtype='uint8') * 240
cv2.putText(img, 'AZ-0001', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2, cv2.LINE_AA)
_, buf = cv2.imencode('.png', img)
img_bytes = buf.tobytes()

# Upload via multipart
boundary = 'testboundary99'
crlf = b'\r\n'
body = b''
body += b'--' + boundary.encode() + crlf
body += b'Content-Disposition: form-data; name="file"; filename="test_query.png"' + crlf
body += b'Content-Type: image/png' + crlf + crlf
body += img_bytes + crlf
body += b'--' + boundary.encode() + b'--' + crlf

req = urllib.request.Request(
    'http://localhost:5000/api/verify',
    data=body,
    headers={'Content-Type': 'multipart/form-data; boundary=' + boundary},
    method='POST'
)
resp = urllib.request.urlopen(req)
result = json.loads(resp.read())
print('Verifikasi berhasil :', result.get('success'))
print('Prediksi penulis    :', result.get('predicted_name'))
print('Similarity          :', result.get('similarity_percent'), '%')
print('Status              :', result.get('verification_status'))
print('Top 3 kandidat:')
for m in result.get('top_matches', [])[:3]:
    print(f"  {m['name']}: {m['percent']}%")
