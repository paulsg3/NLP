# Necessary imports
import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from random import randint
from time import sleep

# Defining necessary functions


def get_ratings(links):
    ratings_list = []
    for link in links:
        article = link.find('article')
        section = article.find('section')
        rating = section.find('div', {'class': 'styles_reviewHeader__iU9Px'})
        rating_num = rating['data-service-review-rating']
        ratings_list.append(rating_num)
    return ratings_list


def get_dates(links):
    dates_list = []
    for link in links:
        article = link.find('article')
        section = article.find('section')
        rating = section.find('div', {'class': 'styles_reviewHeader__iU9Px'})
        date_initial = rating.find('div', {
                                   'class': 'typography_body-m__xgxZ_ typography_appearance-subtle__8_H2l styles_datesWrapper__RCEKH'})
        date = date_initial.find('time')['datetime']
        dates_list.append(date)
    return dates_list


def get_review_body(links):
    reviews_list = []
    for link in links:
        article = link.find('article')
        section = article.find('section')
        review = section.find('div', {'class': 'styles_reviewContent__0Q2Tg'})
        review_text = review.find(
            'p', {'class': 'typography_body-l__KUYFJ typography_appearance-default__AAY17 typography_color-black__5LYEn'})
        if review_text is not None:
            review_text_final = review_text.text
        else:
            review_text_final = None
        reviews_list.append(review_text_final)
    return reviews_list


# Creating dataframe where I will append all info
review_df = pd.DataFrame(
    columns=['Company_Name', 'Date_Published', 'Rating_Value', 'Review_Body'])

# Creating loop to get all info
for val in range(1, 28):
    resp = requests.get(
        'https://www.trustpilot.com/review/www.shopify.com?page={}'.format(val))
    sleep(randint(1, 5))
    html_code = resp.text
    soup = BeautifulSoup(html_code, 'html.parser')
    links = soup.find_all('div', {
                          'class': 'styles_cardWrapper__LcCPA styles_show__HUXRb styles_reviewCard__9HxJJ'})
    ratings_list = get_ratings(links)
    dates_list = get_dates(links)
    reviews_list = get_review_body(links)
    reviews_dict = {'Company_Name': 'Shopify',
                    'Date_Published': dates_list,
                    'Rating_Value': ratings_list,
                    'Review_Body': reviews_list}
    review_df_inter = pd.DataFrame(reviews_dict)
    review_df = pd.concat([review_df, review_df_inter])

# Resetting index
review_df.reset_index(inplace=True, drop=True)

# Getting total reviews
total_reviews = soup.find(
    'p', {'class': 'typography_body-l__KUYFJ typography_appearance-default__AAY17'}).text

cwd = os.getcwd()
path = cwd + "/Reviews.csv"
review_df.to_csv(path, index=False)

print('\n')
print('Dataframe preview: \n')
print(review_df)
print('\n')
print('Total reviews: {}'.format(total_reviews))
print('\n')
