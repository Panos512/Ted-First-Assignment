from os import path
from wordcloud import WordCloud
import pandas as pd


def generate_worldcloud(text):
    """Generates a wordcloud for a given text."""
    # Generate wordcloud
    wordcloud = WordCloud().generate(text)

    # Print wordcloud
    image = wordcloud.to_image()
    image.show()


# Read file
d = path.dirname(__file__)
train_set_path = path.join(d, './data_sets/train_set.csv')
df = pd.read_csv(train_set_path,sep='\t')
for category in df.Category.unique():
    text = df.loc[df['Category'] == category]
    generate_worldcloud(text.to_string())

#text = open(train_set_path).read()
#generate_worldcloud(text)
