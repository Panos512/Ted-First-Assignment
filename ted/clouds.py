from os import path
from wordcloud import WordCloud
import pandas as pd
from tools import open_csv

def generate_worldcloud(text):
    """Generates a wordcloud for a given text."""
    # Generate wordcloud
    wordcloud = WordCloud().generate(text)

    # Print wordcloud
    image = wordcloud.to_image()
    image.show()

df = open_csv()
for category in df.Category.unique():
    text = df.loc[df['Category'] == category]
    generate_worldcloud(text.to_string())
