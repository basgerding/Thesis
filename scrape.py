from urllib.request import urlopen
import time
from bs4 import BeautifulSoup
import csv
import itertools

def main():
    star_list = []
    text_list = []
    url = "https://www.belsimpel.nl/xiaomi-black-shark/reviews"
    page = urlopen(url)
    soup = BeautifulSoup(page, 'html.parser')
    star_rating_box = soup.find_all('span', attrs = {'class': 'rs_stars_score'})
    review_text_box = soup.find_all('div', attrs = {'class': 'user_review_text'})
    for stars in star_rating_box:
        star_rating = stars.text.split()[0]
        star_list.append(star_rating)
    for texts in review_text_box:
        review_text = " ".join(texts.text.split())
        text_list.append(review_text)
    with open('reviews.csv', 'a+') as csv_file:
        writer = csv.writer(csv_file)
        for reviews, stars in zip(text_list, star_list[2:-4]):
            writer.writerow([reviews+"|"+stars])

    time.sleep(1)

if __name__ == '__main__':
    main()
