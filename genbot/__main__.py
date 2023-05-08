import json
import logging

from torch.nn import BCEWithLogitsLoss
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import IntentClassificationDataset, IntentPredictionDataset
from models import IntentClassifier, IntentPredictor

DATASET_FILENAME = './data/clean/customer_support_twitter_sample.json'
TESTSET_FILENAME = './data/clean/customer_support_twitter_sample_test.json'
N_EPOCHS = 2
DEVICE = 'cuda'

logging.basicConfig(level=logging.INFO, format='[{levelname}] {message}', style='{')


def get_classifier_dataset() -> IntentClassificationDataset:
    with open(DATASET_FILENAME) as file:
        data = json.load(file)
    return IntentClassificationDataset(data)


def init_classifier(dataset: IntentClassificationDataset) -> IntentClassifier:
    return IntentClassifier(dataset.n_labels, optimizer_class=Adam,
                            criterion_class=BCEWithLogitsLoss).to(DEVICE)


def train_intent_classifier(classifier: IntentClassifier, dataset: IntentClassificationDataset) -> None:
    classifier.train()
    for epoch in tqdm(range(N_EPOCHS)):
        logging.info(f"Epoch {epoch}")
        running_loss = 0.
        for inputs, targets in DataLoader(dataset, batch_size=2):
            outputs = classifier(inputs)
            targets = targets.to(DEVICE)
            loss = classifier.criterion(outputs, targets)
            running_loss += loss
            loss.backward()
            classifier.optimizer.step()
        logging.info(f'running_loss: {running_loss / len(dataset)}')


def get_classifier_testset(intents) -> IntentClassificationDataset:
    with open(TESTSET_FILENAME) as file:
        test = json.load(file)
    return IntentClassificationDataset(test, intents)


def evaluate_intent_classifier(classifier, dataset: IntentClassificationDataset):
    classifier.eval()
    losses = []
    for inputs, targets in DataLoader(dataset, batch_size=2):
        outputs = classifier(inputs)
        targets = targets.to('cuda')
        loss = classifier.criterion(outputs, targets)
        losses.append(loss.item())
    return sum(losses) / len(losses)


def get_predictor_dataset() -> IntentPredictionDataset:
    with open(DATASET_FILENAME) as file:
        data = json.load(file)
    return IntentPredictionDataset(data)


def get_predictor_testset() -> IntentPredictionDataset:
    with open(DATASET_FILENAME) as file:
        data = json.load(file)
    return IntentPredictionDataset(data)


def init_predictor() -> IntentPredictor:
    return IntentPredictor().to('cuda')


def train_intent_predictor(predictor: IntentPredictor, dataset: IntentPredictionDataset) -> None:
    predictor.train()
    for epoch in tqdm(range(N_EPOCHS)):
        logging.info(f"Epoch {epoch}")
        running_loss = 0.
        for inputs, targets in DataLoader(dataset, batch_size=2):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = predictor(inputs)
            loss = predictor.criterion(outputs, targets)
            running_loss += loss
            loss.backward()
            predictor.optimizer.step()
        logging.info(f'running_loss: {running_loss / len(dataset)}')


def evaluate_intent_predictor(predictor: IntentPredictor, dataset: IntentPredictionDataset) -> float:
    predictor.eval()
    losses = []
    for inputs, targets in DataLoader(dataset, batch_size=2):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = predictor(inputs)
        loss = predictor.criterion(outputs, targets)
        losses.append(loss.item())
    return sum(losses) / len(losses)


def main():
    logging.info('Starting GenBot')
    # # Setup classifier datasets
    # classifier_dataset = get_classifier_dataset()
    # classifier_testset = get_classifier_testset(classifier_dataset.intents)
    # # Setup, train, & evaluate classifier
    # classifier = init_classifier(classifier_dataset)
    # train_intent_classifier(classifier, classifier_dataset)
    # loss = evaluate_intent_classifier(classifier, classifier_testset)
    # logging.info(f'Evaluation loss: {loss}')

    # # Setup intent predictor datasets
    # predictor_dataset = get_predictor_dataset()
    # predictor_testset = get_predictor_testset()
    # # Setup, train, & evaluate intent predictor
    # predictor = init_predictor()
    # train_intent_predictor(predictor, predictor_dataset)
    # loss = evaluate_intent_predictor(predictor, predictor_testset)
    # logging.info(f'Evaluation loss: {loss}')




if __name__ == "__main__":
    main()
