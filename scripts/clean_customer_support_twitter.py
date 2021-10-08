import json
import re
from typing import List, Dict

import pandas as pd
from npyscreen import NPSApp, TitleMultiSelect, Form, TitleText, NPSAppManaged

CORPUS_FILE = '../data/raw/customer_support_twitter_sample.csv'
OUTPUT_FILE = '../data/clean/customer_support_twitter_sample.json'
TARGET_COMPANY = 'AppleSupport'


def load_dataframe(corpus_file_name) -> pd.DataFrame:
    return pd.read_csv(CORPUS_FILE)


def get_company_tweets_dataframe(dataframe, target_company) -> pd.DataFrame:
    return dataframe[dataframe['author_id'] == target_company]


def get_related_tweets_dataframe(dataframe, company_tweets_dataframe) -> pd.DataFrame:
    ids = set()
    for tweet_id, _, _, _, _, response_tweet_id, in_response_to_tweet_id in company_tweets_dataframe.values:
        ids.add(str(tweet_id))  # Add company tweet
        ids.add(str(int(in_response_to_tweet_id)))  # Add parent tweet
        if type(response_tweet_id) == str:  # Add all direct children tweets
            for response_id in response_tweet_id.split(','):
                ids.add(response_id)
    return dataframe[dataframe['tweet_id'].isin(ids)]


def get_first_tweets_dataframe(dataframe) -> pd.DataFrame:
    return dataframe[dataframe['in_response_to_tweet_id'].isnull()]


def get_last_tweets_dataframe(dataframe) -> pd.DataFrame:
    return dataframe[dataframe['response_tweet_id'].isnull()]


def get_previous_tweet(dataframe, tweet):
    try:
        return dataframe[dataframe['response_tweet_id'] == str(tweet[0])].iloc[0]
    except IndexError:
        pass
    return None


def to_dict(tweet):
    return {
        'text': tweet.text,
        'authored': not tweet.inbound,
        'intent': None
    }


class IntentTypeForm(Form):
    DEFAULT_LINES = 17

    def create(self):
        tweet = self.parentApp.tweet
        self.author = self.add(TitleText, name='Author', editable=False,
                               value=str(tweet.get('authored')))
        self.text = self.add(TitleText, name='Text', editable=False, value=tweet.get('text'))
        self.intent = self.add(TitleText, name='Intent')

    def beforeEditing(self):
        tweet = self.parentApp.tweet
        self.text.set_value(tweet.get('text'))
        self.intent.set_value("")

    def afterEditing(self):
        if self.intent:
            self.parentApp.intents.add(self.intent.value)
            self.parentApp.setNextForm('select')


class IntentSelectForm(Form):
    DEFAULT_LINES = 17

    def create(self):
        tweet = self.parentApp.tweet
        self.help_text = self.add(TitleText, name='Help', editable=False,
                                  value='Select one or more of the intents listed. If the intent is not there, select none and you can enter 1 manually',
                                  w_id='help_text')
        self.author = self.add(TitleText, name='Author', editable=False,
                               value='Company' if tweet.get('authored') else 'User')
        self.text = self.add(TitleText, name='Text', editable=False, value=tweet.get('text'), w_id='text')
        intents = list(self.parentApp.intents)
        self.intent_selection = self.add(TitleMultiSelect, value=[], name="Select intent(s)", values=intents,
                                         scroll_exit=True)

    def beforeEditing(self):
        tweet = self.parentApp.tweet
        self.text.set_value(tweet.get('text'))
        intents = list(self.parentApp.intents)
        intents.sort()
        self.intent_selection.set_values(intents)
        self.intent_selection.set_value(list())

    def afterEditing(self):
        selected_intents = self.intent_selection.get_selected_objects()
        if selected_intents:
            self.parentApp.setNextForm('select')
            self.parentApp.tweet['intent'] = '+'.join(selected_intents)
        else:
            self.parentApp.setNextForm('create')


class IntentApp(NPSAppManaged):
    STARTING_FORM = "select"

    def __init__(self, conversations: list):
        self.conversations = conversations
        self.tweet_generator = self.next_tweet()
        self.tweet = next(self.tweet_generator)
        self.intents = set()
        super().__init__()

    def onStart(self):
        self.addForm('select', IntentSelectForm, name='Intent Select Form')
        self.addForm('create', IntentTypeForm, name='Intent Create Form')

    def onInMainLoop(self):
        print(self.getForm('select').text.value)
        print(self.tweet, type(self.tweet))
        if self.NEXT_ACTIVE_FORM == 'select' and self.tweet.get('intent'):
            self.tweet = next(self.tweet_generator)
            if not self.tweet:
                self.setNextForm(None)

    def next_tweet(self):
        for conversation in self.conversations:
            for tweet in conversation:
                yield tweet
        yield None


def replace_urls(conversations: List[List[Dict]]):
    URL_RE = r'(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})'
    for conversation in conversations:
        for tweet in conversation:
            tweet['text'] = re.sub(URL_RE, 'URL', tweet.get('text', ''))


def replace_usernames(conversations: List[List[Dict]]):
    USER_RE = r'@\d+'
    for conversation in conversations:
        for tweet in conversation:
            tweet['text'] = re.sub(USER_RE, 'USERNAME', tweet.get('text', ''))


def replace_unicode(conversations: List[List[Dict]]):
    unicode_replacements = {
        '\u2019': '\''
    }
    for conversation in conversations:
        for tweet in conversation:
            for unicode, replacement in unicode_replacements.items():
                tweet['text'] = tweet.get('text').replace(unicode, replacement)
            tweet['text'] = tweet.get('text').encode('ascii', 'ignore').decode()


def replace_html(conversations: List[List[Dict]]):
    """
    Replaces things like '$gt;' with '>'
    :param conversations:
    :return:
    """
    from html import unescape
    for conversation in conversations:
        for tweet in conversation:
            tweet['text'] = unescape(tweet.get('text'))


if __name__ == '__main__':
    # df = load_dataframe(CORPUS_FILE)
    # company_tweets_df = get_company_tweets_dataframe(df, TARGET_COMPANY)
    # related_tweets_df = get_related_tweets_dataframe(df, company_tweets_df)
    # last_tweets_df = get_last_tweets_dataframe(related_tweets_df)
    # data = []
    # for _, last_tweet in last_tweets_df.iterrows():
    #     conversation = [to_dict(last_tweet)]
    #     previous_tweet = get_previous_tweet(df, last_tweet)
    #     while previous_tweet is not None:
    #         conversation.append(to_dict(previous_tweet))
    #         previous_tweet = get_previous_tweet(df, previous_tweet)
    #     conversation.reverse()
    #     data.append(conversation)
    #
    # app = IntentApp(data)
    # app.run()
    #
    #
    data = json.load(open(OUTPUT_FILE, 'r'))

    replace_urls(data)
    replace_usernames(data)
    replace_unicode(data)
    replace_html(data)

    json.dump(data, open(OUTPUT_FILE, 'w'), indent=4)

    for c in data:
        print(c)
