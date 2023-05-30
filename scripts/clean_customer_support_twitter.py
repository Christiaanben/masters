import json
import logging
import re
from typing import List, Dict, Optional, Set

import click
import inquirer
import pandas as pd
from tqdm.auto import tqdm

CORPUS_FILE = '../data/raw/customer_support_twitter_full.csv'
OUTPUT_FILE = '../data/clean/customer_support_twitter_full.json'
TARGET_COMPANY = 'AppleSupport'
N_CONVERSATIONS = 100
UTTER = 'utter_'


def _load_dataframe(file_path) -> pd.DataFrame:
    logging.debug(f'Loading dataframe from {file_path}...')
    df = pd.read_csv(file_path)
    logging.debug(f"Loaded dataframe with shape {df.shape}")
    return df


def _load_conversations(file_path) -> List[List[Dict]]:
    logging.debug(f'Loading conversations from {file_path}...')
    with open(file_path, 'r') as f:
        conversations = json.load(f)
        logging.debug(f"Loaded {len(conversations)} conversations")
        return conversations


def _get_company_tweets(df: pd.DataFrame, target_company: str) -> pd.DataFrame:
    company_tweets_df = df[df['author_id'] == target_company]
    logging.debug(f"Found {len(company_tweets_df)} tweets from {target_company}")
    return company_tweets_df


def _get_related_tweets(df: pd.DataFrame, company_tweets_df: pd.DataFrame) -> pd.DataFrame:
    ids = set()
    for tweet_id, _, _, _, _, response_tweet_id, in_response_to_tweet_id in company_tweets_df.values:
        try:
            ids.add(tweet_id)  # Add company tweet
            if float.is_integer(in_response_to_tweet_id):
                ids.add(int(in_response_to_tweet_id))  # Add parent tweet
            if type(response_tweet_id) == str:  # Add all direct children tweets
                response_tweet_ids = response_tweet_id.split(',')
                for response_id in response_tweet_id.split(','):
                    ids.add(response_id)
        except ValueError as ex:
            logging.error(f"Could not parse tweet id {tweet_id}")
            logging.debug(
                f'Tweet ID: {tweet_id}, in_response_to_tweet_id: {in_response_to_tweet_id}, response_tweet_id: {response_tweet_id}')
            raise ex
    return df[df['tweet_id'].isin(ids)]


def _get_last_tweets(df: pd.DataFrame) -> pd.DataFrame:
    return df[df['response_tweet_id'].isnull()]


def _to_dict(tweet):
    return {
        'id': int(tweet.tweet_id),
        'text': tweet.text,
        'authored': not tweet.inbound,
        'intents': []
    }


def _get_previous_tweet(df: pd.DataFrame, tweet: pd.Series) -> Optional[pd.Series]:
    try:
        return df[df['response_tweet_id'] == str(tweet[0])].iloc[0]
    except IndexError:
        return None


def _get_conversations(dataframe: pd.DataFrame, target_company: str) -> List[List[Dict]]:
    logging.debug('Extracting conversations...')
    company_tweets_df = _get_company_tweets(dataframe, target_company)
    related_tweets_df = _get_related_tweets(dataframe, company_tweets_df)
    last_tweets_df = _get_last_tweets(related_tweets_df)
    conversations = []
    for _, last_tweet in tqdm(last_tweets_df.iterrows(), total=len(last_tweets_df)):
        conversation = [_to_dict(last_tweet)]
        previous_tweet = _get_previous_tweet(related_tweets_df, last_tweet)
        while previous_tweet is not None:
            conversation = [_to_dict(previous_tweet)] + conversation
            previous_tweet = _get_previous_tweet(related_tweets_df, previous_tweet)
        conversations.append(conversation)
        if N_CONVERSATIONS and len(conversations) > N_CONVERSATIONS:
            break
    logging.debug(f"Extracted {len(conversations)} conversations")
    return conversations


def _get_all_intents(conversations: List[List[Dict]]) -> Set[str]:
    intents = set()
    for conversation in conversations:
        for message in conversation:
            for intent in message['intents']:
                intents.add(intent)
    return intents


def _ask_intents(message: Dict, all_intents: Set[str]) -> List[str]:
    other = 'Add intent'
    is_authored = message['authored']
    if is_authored:
        intents = [intent for intent in all_intents if intent.startswith(UTTER)]
    else:
        intents = [intent for intent in all_intents if not intent.startswith(UTTER)]
    choices = list(intents) + [other]
    print(message['text'])
    question = f'Select intents for message:'
    questions = [inquirer.Checkbox('intents', message=question, choices=choices, validate=lambda _, x: x != [])]
    answers = inquirer.prompt(questions)
    if not answers:
        raise KeyboardInterrupt
    if other in answers['intents']:
        questions = [inquirer.Text(
            'new_intent',
            message='Which intent would you like to add?',
            validate=lambda _, x: x != '')]
        answers = inquirer.prompt(questions)
        all_intents.add(answers['new_intent'])
        return _ask_intents(message, all_intents)
    return answers['intents']


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
        '\u2019': "'"
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


def _assign_labels(conversations: List[List[Dict]]) -> List[List[Dict]]:
    all_intents = _get_all_intents(conversations)
    try:
        for conversation in conversations:
            for message in conversation:
                if not message['intents']:
                    intents = _ask_intents(message, all_intents)
                    message['intents'] = intents
                    all_intents.update(intents)
    except KeyboardInterrupt:
        logging.warning('Interrupted by user. Saving progress...')
    return conversations


def _save_conversations(conversations: List[List[Dict]], output_file: str):
    logging.debug(f'Saving conversations to {output_file}...')
    with open(output_file, 'w') as f:
        json.dump(conversations, f, indent=2)
    logging.debug('Saved conversations')


@click.command()
@click.option('--corpus-file', default=CORPUS_FILE, help='Path to corpus file.', type=click.Path(exists=True))
@click.option('--output-file', default=OUTPUT_FILE, help='Path to output file.', type=click.Path())
@click.option('--target_company', default=TARGET_COMPANY, help='Target company to extract tweets from.')
@click.option('--continue/--no-continue', 'continue_', default=True, help='Continue from last saved progress.')
def main(corpus_file, output_file, target_company, continue_):
    logging.info("Starting cleaning process...")
    if continue_:
        conversations = _load_conversations(output_file)
    else:
        df = _load_dataframe(corpus_file)
        conversations = _get_conversations(df, target_company)
    replace_urls(conversations)
    replace_usernames(conversations)
    replace_unicode(conversations)
    replace_html(conversations)
    conversations = _assign_labels(conversations)
    _save_conversations(conversations, output_file)


def setup_logging():
    logging.basicConfig(level=logging.DEBUG, format='[{levelname}] {message}', style='{')


if __name__ == '__main__':
    setup_logging()
    main()
