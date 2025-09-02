import requests

def test_prediction(text: str):
    url = "http://127.0.0.1:8010/predict"
    data = {"text": text}
    response = requests.post(url, json=data)
    print("Response JSON:", response.json())
    return response.json()['prediction']

if __name__ == "__main__":
    test_text1 = "还是很不错的"
    print(test_text1)
    print(test_prediction(test_text1))
    test_text2="虽然在某些细节方面有点问题，特别是包装材质上，但总体来说还是不错的"
    print(test_text2)
    print(test_prediction(test_text2))
    test_text3= "#哪吒联名周边# 我宣布 ！！！本细节控超满意！！！🥰霸王茶姬你搞谷子是真的有点东西，每个周边都深得我心👌🏻💖blingbling的✨ 🤪@藤ONE-瀚墨 快来和我一起参与茶姬抢座，赢取哪吒2联名周边吧！👌🤪#霸王茶姬联名哪吒2燃起来了#"
    print(test_text3)
    print(test_prediction(test_text3))
