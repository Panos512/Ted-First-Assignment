from os import path
from wordcloud import WordCloud
import pandas as pd
from tools import open_csv

def generate_worldcloud(text, category):
    """Generates a wordcloud for a given text."""
    # Generate wordcloud
    wordcloud = WordCloud().generate(text)

    # Print wordcloud
    image = wordcloud.to_image()
    image.save('./ted/outputs/'+category+'_cloud.bmp')
    #image.show()

df = open_csv()
for category in df.Category.unique():
    text = df.loc[df['Category'] == category]
    generate_worldcloud(text.Content.to_string(), category)
