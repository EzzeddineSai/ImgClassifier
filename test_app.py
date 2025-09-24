import requests

ENDPONT = "http://localhost:8000"

def test_home():
    address = ENDPONT + "/"
    response = requests.get(address)
    assert response.status_code == 200
    pass

def test_uploadfile_status():
    address = ENDPONT + "/uploadfile"
    files = {'file': open('./test_data/frog.jpg', 'rb')}
    response = requests.post(address, files=files)
    assert response.status_code == 200
    pass

def test_uploadfile_status():
    address = ENDPONT + "/uploadfile"
    files = {'file': open('./test_data/frog.jpg', 'rb')}
    response = requests.post(address, files=files)
    assert response.status_code == 200
    print(response.json())
    pass