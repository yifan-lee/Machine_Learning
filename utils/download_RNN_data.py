import yfinance as yf
import pandas as pd

# 要下载的金融产品列表（股票、指数、ETF都可以）
tickers = [
    "FXAIX",   # 富达500指数基金
    "FSKAX",   # 富达全市场指数基金
    "FSPSX",   # 富达国际指数基金
    "MSFT",    # 微软
    "AAPL",    # 苹果
    "TSLA",    # 特斯拉
    "MCD",     # 麦当劳
    "BRK-B",   # 伯克希尔哈撒韦B
    "GOOG",    # 谷歌
    "AXP",     # 美国运通 (Amex)
    "JPM",     # 摩根大通
    "KHC",     # 卡夫亨氏
    "SBUX",    # 星巴克
    "AMZN",    # 亚马逊
    "KO",      # 可口可乐
    "NKE",     # 耐克
    "VZ",      # Verizon
    "NVDA",    # 英伟达
    "PG",      # 宝洁
    "PEP",     # 百事可乐
    "COST",    # Costco
    "^IRX",    # 美国3个月短期国债 (注意：yfinance中可能没有TB3M, 需要用^IRX)
    "ASML",    # ASML
    "BJ",      # BJ's Wholesale Club
    "C",       # 花旗银行
    "COF",     # Capital One
    "FRCOY",   # 法国农业信贷海外存托
    "LOGI",    # Logitech
    "META",    # Meta (Facebook)
    "NTDOY",   # Nintendo
    "PFE",     # Pfizer
    "RYKKY",   # Ryohin Keikaku (Muji)
    "SONY",    # Sony
    "SPG",     # Simon Property Group
    "TGT",     # Target
    "^GSPC",   # 标普500指数
    "^NDX",    # 纳斯达克100指数
    "GLD",     # 黄金ETF
    "TLT",     # 美国长期国债ETF
]

# 下载2020-01-01到2025-07-30的日线数据
data = yf.download(
    tickers,
    start="2020-01-01",
    end="2025-01-01",
    group_by="ticker"  # 方便区分不同资产
)

# 保存为CSV
data.to_csv("./data/RNN/financial_timeseries.csv")
print("数据下载完成，保存为 ./data/financial_timeseries.csv")




import pandas as pd
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 参数
num_words = 10000  # 只保留最常见的 10000 个词
maxlen = 200       # 句子统一长度，超过截断，不足补0

# 1. 下载数据
(xTrain, yTrain), (xTest, yTest) = imdb.load_data(num_words=num_words)

# 2. 把索引序列转成统一长度
xTrain = pad_sequences(xTrain, maxlen=maxlen)
xTest = pad_sequences(xTest, maxlen=maxlen)

# 3. 获取词典，索引到单词的映射
wordIndex = imdb.get_word_index()
indexWord = {v+3: k for k, v in wordIndex.items()}
indexWord[0] = "<PAD>"
indexWord[1] = "<START>"
indexWord[2] = "<UNK>"
indexWord[3] = "<UNUSED>"

# 4. 把索引序列转成句子
def decode_review(sequence):
    return " ".join([indexWord.get(i, "?") for i in sequence])

trainTexts = [decode_review(seq) for seq in xTrain]
testTexts = [decode_review(seq) for seq in xTest]

# 5. 保存为 CSV
dfTrain = pd.DataFrame({"text": trainTexts, "label": yTrain})
dfTest = pd.DataFrame({"text": testTexts, "label": yTest})

dfTrain.to_csv("./data/RNN/IMDB/imdb_train.csv", index=False)
dfTest.to_csv("./data/RNN/IMDB/imdb_test.csv", index=False)

print("IMDB 数据集已保存为 imdb_train.csv 和 imdb_test.csv")