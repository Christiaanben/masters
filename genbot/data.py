import pandas as pd
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data.dataset import T_co


def get_subset_for_company(df, company: str) -> pd.DataFrame:
    ids = set()
    company_tweets = df[df['author_id'] == company]
    print(company_tweets)
    for tweet_id, author_id, inbound, created_at, text, response_tweet_id, in_response_to_tweet_id in company_tweets.values:
        ids.add(str(tweet_id))
        print(tweet_id)
        ids.add(str(int(in_response_to_tweet_id)))
        if type(response_tweet_id) == str:
            for response_id in response_tweet_id.split(','):
                ids.add(response_id)
    print(ids)
    return df[df['tweet_id'].isin(ids)]


class Dataset(TorchDataset):

    def __init__(self, df: pd.DataFrame, company='AppleSupport'):
        self.df = get_subset_for_company(df, company)

    def __getitem__(self, index) -> T_co:
        return None

    def __len__(self) -> int:
        return 0


if __name__ == '__main__':
    df = pd.read_csv('../sample.csv')
    dataset = Dataset(df)
    print(dataset.df.values)
    print(len(dataset))

