from requests import get
from bs4 import BeautifulSoup as soup
from random import randint
from time import sleep
import os
import sys
import inspect
import time
from nltk.corpus import stopwords
from stop_words import get_stop_words
from nltk.tokenize import RegexpTokenizer
import csv

# Define a list of stop words
stop_words = list(get_stop_words('en'))  # About 900 stopwords
nltk_words = list(stopwords.words('english'))  # About 150 stopwords
punctuations = [".", "'", ","]
stop_words.extend(nltk_words + punctuations)

# Process each sentence to extract relevant information
processed_review = []

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # noqa: E501
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

start_time = time.time()

# This loop scrapes the first 10 pages of each hotel from the below "url"
hotel_pages = ["", "oa30-", "oa60-", "oa90-", "oa120-", "oa150-", "oa180-", "oa210-", "oa240-", "oa270-"]  # noqa: E501

reviews = []
churn = []
data_list = []
for page in hotel_pages:

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.67 Safari/537.36 OPR/87.0.4390.45'}  # noqa: E501
    url = 'https://www.tripadvisor.in/Hotels-g30196-{pageKey}Austin_Texas-Hotels.html'.format(pageKey=page)  # noqa: E501

    html = get(url, headers=headers, timeout=5, allow_redirects=True)
    bsobj = soup(html.content, 'lxml')
    links = []

    review_pages = [""]  # , "-or10", "-or20", "30", "40", "50", "60", "70", "80", "90"]  # noqa: E501
    # This loop scrapes the first 10 pages of reviews for each hotel
    for review in bsobj.findAll('a', {'class': 'review_count'}):
        for page_suffix in review_pages:
            a = review['href']
            a = 'https://www.tripadvisor.in' + a
            a = a[:(a.find('Reviews')+7)] + '{pageKey1}'.format(pageKey1=page_suffix) + a[(a.find('Reviews')+7):]  # noqa: E501
            links.append(a)

    print(len(links))

    for link in links:
        d = [5, 10, 15, 20, 25]
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36'}  # noqa: E501
        html2 = get(link.format(i for i in range(5, 1000, 5)), headers=headers)
        sleep(randint(1, 5))
        bsobj2 = soup(html2.content, 'lxml')

        hotelName = bsobj2.find(id='HEADING')

        print('*****')
        print(link)
        print('*****')
        for title, reviewWithReadmore, date_stayed, trip_type, rating in zip(bsobj2.findAll(class_='Qwuub'), bsobj2.findAll(class_='QewHA H4 _a'), bsobj2.findAll(class_='teHYY _R Me S4 H3'), bsobj2.findAll(class_="TDKzw _R Me"), bsobj2.findAll('div', class_='Hlmiy F1')):  # noqa: E501

            print(hotelName.text)

            reviewWithReadmoreAndTitle = " ".join((title.span.text.strip(), reviewWithReadmore.span.text.strip()))  # noqa: E501
            tokenizer = RegexpTokenizer(r'\w+')
            word_list = tokenizer.tokenize(reviewWithReadmoreAndTitle.lower())
            output = [w for w in word_list if not w in stop_words]  # noqa: E501
            processed_review = ' '.join(word_list)
            print(processed_review)

            processed_date_stayed = date_stayed.text[14:]
            print(processed_date_stayed)

            processed_trip_type = trip_type.text[21:]
            print(processed_trip_type)

            rating_span = rating.find('span', class_='ui_bubble_rating')
            processed_rating = rating_span.attrs['class'][1].replace('bubble_', '')  # noqa: E501
            print(processed_rating[:1])
            print('\n')
            # Append the data to the list
            data_list.append({'hotel_name': hotelName.text, 'processed_review': processed_review,  # noqa: E501
                              'date_stayed': processed_date_stayed, 'trip_type': processed_trip_type,  # noqa: E501
                              'processed_rating': processed_rating[:1]})  # noqa: E501

with open('reviews.csv', 'a', encoding="utf-8") as csv_file:
    fieldnames = ['hotel_name', 'processed_review', 'date_stayed', 'trip_type', 'processed_rating']  # noqa: E501
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator='\n')  # noqa: E501
    csv_writer.writeheader()
    for data in data_list:
        csv_writer.writerow(data)

print("time taken: " + str((time.time() - start_time)))
