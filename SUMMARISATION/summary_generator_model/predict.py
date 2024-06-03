import transformers
from transformers import pipeline
from data_loader import small_cnn_dailymail

"""TEST ON EXAMPLE"""

summarizer = pipeline("summarization", model="t5_small_spoilers")

test_text = small_cnn_dailymail["test"]["article"][:1]
print(summarizer(test_text))

text = " Green tea is more than just a soothing beverage; it's packed with health benefits that make it a smart choice for daily consumption. Firstly, green tea is rich in antioxidants, particularly catechins, which help to fight inflammation and protect cells from damage. This can lower the risk of chronic diseases like heart disease and certain cancers. Moreover, green tea contains compounds that may boost metabolism and promote fat loss, making it a valuable tool for weight management when combined with a healthy diet and exercise. Additionally, green tea has been linked to improved brain function and a reduced risk of cognitive decline with aging. The combination of caffeine and L-theanine in green tea can enhance alertness and focus without the jittery side effects associated with coffee. Lastly, green tea may support overall longevity and promote healthy aging thanks to its protective effects on various aspects of health. Incorporating green tea into your daily routine can be a simple yet effective way to support your health and well-being."

print(summarizer(text))