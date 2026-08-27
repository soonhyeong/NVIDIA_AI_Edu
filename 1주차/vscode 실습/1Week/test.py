from io import StringIO
import matplotlib.pyplot as plt
import pandas as pd
import requests

url = "https://ko.wikipedia.org/wiki/%EA%B8%B0%EB%8C%80%EC%88%98%EB%AA%85%EC%88%9C_%EB%82%98%EB%9D%BC_%EB%AA%A9%EB%A1%9D"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0 Safari/537.36"
}

response = requests.get(url, headers=headers)

df_list = pd.read_html(StringIO(response.text))
df = df_list[0]
print(df.head())
