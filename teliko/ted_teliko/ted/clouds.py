import string

from wordcloud import WordCloud
from sklearn.feature_extraction import text

from tools import open_csv


def generate_worldcloud(text, category):
    """Generates a wordcloud for a given text."""
    # Generate wordcloud
    wordcloud = WordCloud().generate(text)

    # Print wordcloud
    image = wordcloud.to_image()
    image.save('./outputs/'+category+'_cloud.bmp')

df = open_csv()
for category in df.Category.unique():
    result = df.loc[df['Category'] == category]

    stop_words = text.ENGLISH_STOP_WORDS

    # Getting the content of each category
    words = result.Content
    # removing stopwords
    words = words.apply(lambda x: ' '.join([word for word in string.split(x) if word.lower() not in stop_words]))

    generate_worldcloud(words.to_string(), category)
