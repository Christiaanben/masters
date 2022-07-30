import json

from torch.nn import BCEWithLogitsLoss
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from genbot.data import IntentClassificationDataset
from genbot.model import IntentClassifier

DATASET_FILENAME = '../data/clean/customer_support_twitter_sample.json'
TESTSET_FILENAME = '../data/clean/customer_support_twitter_sample_test.json'
N_EPOCHS = 2


def main():
    with open(DATASET_FILENAME) as file:
        data = json.load(file)
    classifier_dataset = IntentClassificationDataset(data)
    classifier = IntentClassifier(classifier_dataset.n_labels, optimizer_class=Adam, criterion_class=BCEWithLogitsLoss).to('cuda')
    classifier.train()
    for epoch in tqdm(range(N_EPOCHS)):
        print("Epoch", epoch)
        running_loss = 0.
        for inputs, targets in DataLoader(classifier_dataset, batch_size=2):
            outputs = classifier(inputs)
            targets = targets.to('cuda')
            loss = classifier.criterion(outputs, targets)
            running_loss += loss
            loss.backward()
            classifier.optimizer.step()
        print('running_loss:', running_loss/len(classifier_dataset))

    with open(TESTSET_FILENAME) as file:
        test = json.load(file)
    classifier_testset = IntentClassificationDataset(test, classifier_dataset.intents)
    classifier.eval()

    for inputs, targets in DataLoader(classifier_testset, batch_size=2):
        outputs = classifier(inputs)
        targets = targets.to('cuda')
        loss = classifier.criterion(outputs, targets)
        print(loss)


if __name__ == "__main__":
    main()
