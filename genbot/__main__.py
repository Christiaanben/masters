import json
import logging

import torch
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from torchmetrics.functional.classification import multilabel_f1_score
from tqdm import tqdm
import lightning.pytorch as pl
from transformers import DistilBertTokenizerFast

from genbot.data import GeneratorDataset, IntentClassificationDataset, IntentPredictionDataset
from genbot.models.generator import Generator
from models import IntentClassifier, IntentPredictor

DATASET_FILENAME = '../data/clean/customer_support_twitter_full.json'
TESTSET_FILENAME = '../data/clean/customer_support_twitter_full_test.json'
N_EPOCHS = 2
RETRAIN = True
DEVICE = 'cuda'

logging.basicConfig(level=logging.INFO, format='[{levelname}] {message}', style='{')


def get_classifier_dataset(tokenizer) -> IntentClassificationDataset:
    with open(DATASET_FILENAME) as file:
        data = json.load(file)
    return IntentClassificationDataset(data, tokenizer=tokenizer)


def init_classifier(dataset: IntentClassificationDataset) -> IntentClassifier:
    return IntentClassifier(dataset.n_labels)



def get_classifier_testset(intents, tokenizer) -> IntentClassificationDataset:
    with open(TESTSET_FILENAME) as file:
        test = json.load(file)
    return IntentClassificationDataset(test, intents, tokenizer=tokenizer)

def get_predictor_dataset() -> IntentPredictionDataset:
    with open(DATASET_FILENAME) as file:
        data = json.load(file)
    return IntentPredictionDataset(data)


def get_predictor_testset() -> IntentPredictionDataset:
    with open(DATASET_FILENAME) as file:
        data = json.load(file)
    return IntentPredictionDataset(data)


def evaluate_intent_predictor(predictor: IntentPredictor, dataset: IntentPredictionDataset) -> float:
    predictor.eval()
    predictor.to(DEVICE)
    losses = []
    for inputs, targets in DataLoader(dataset, batch_size=2):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        outputs = predictor(inputs)
        loss = predictor.criterion(outputs, targets)
        losses.append(loss.item())
    return sum(losses) / len(losses)


def get_generator_dataset() -> GeneratorDataset:
    with open(DATASET_FILENAME) as file:
        data = json.load(file)
    return GeneratorDataset(data, tokenizer=Generator.init_tokenizer())


def init_generator() -> Generator:
    return Generator().to(DEVICE)


def train_generator(generator: Generator, dataset: GeneratorDataset) -> None:
    generator.train()
    try:
        for epoch in tqdm(range(30)):
            logging.info(f"Epoch {epoch}")
            running_loss = 0.
            for inputs, attention_mask, labels in DataLoader(dataset, batch_size=2):
                generator.optimizer.zero_grad()
                inputs, attention_mask, labels = inputs.to(DEVICE), attention_mask.to(DEVICE), labels.to(DEVICE)
                loss = generator(input_ids=inputs, attention_mask=attention_mask, labels=labels).loss
                running_loss += loss.item()
                loss.backward()
                generator.optimizer.step()
            logging.info(f'running_loss: {running_loss / len(dataset)}')
    except KeyboardInterrupt:
        logging.info('Interrupted')


def evaluate_generator(generator: Generator, dataset: GeneratorDataset) -> float:
    generator.eval()
    running_loss = 0.
    for inputs, attention_mask, labels in DataLoader(dataset, batch_size=2):
        inputs, attention_mask, labels = inputs.to(DEVICE), attention_mask.to(DEVICE), labels.to(DEVICE)
        loss = generator(input_ids=inputs, attention_mask=attention_mask, labels=labels).loss
        running_loss += loss.item()
    return running_loss / len(dataset)


def main():
    logging.info('Starting GenBot')
    # Setup classifier datasets
    # classifier_tokenizer = DistilBertTokenizerFast.from_pretrained(IntentClassifier.model_name)
    # classifier_dataset = get_classifier_dataset(classifier_tokenizer)
    # classifier_testset = get_classifier_testset(classifier_dataset.intents, classifier_tokenizer)
    # # Setup, train, & evaluate classifier
    # classifier = init_classifier(classifier_dataset)
    #
    # trainer = pl.Trainer(max_epochs=3)
    # trainer.fit(
    #     classifier,
    #     DataLoader(classifier_dataset, batch_size=2, shuffle=True),
    #     DataLoader(classifier_testset, batch_size=2, shuffle=False)
    # )
    # trainer.validate(classifier, DataLoader(classifier_testset, batch_size=2, shuffle=False))
    # Eval: classifier_dataset.intents[torch.argmax(classifier.forward(classifier_tokenizer(text, return_tensors='pt')))]

    # Setup intent predictor datasets
    predictor_dataset = get_predictor_dataset()
    predictor_testset = get_predictor_testset()
    # Setup, train, & evaluate intent predictor
    predictor = IntentPredictor(n_labels=50)
    predictor_trainer = pl.Trainer(max_epochs=2)
    predictor_trainer.fit(predictor, DataLoader(predictor_dataset, batch_size=2))
    loss = evaluate_intent_predictor(predictor, predictor_testset)
    logging.info(f'Evaluation loss: {loss}')

    # # Setup generator datasets
    # generator_dataset = get_generator_dataset()
    # # Setup, train, & evaluate generator
    # generator = init_generator()
    # train_generator(generator, generator_dataset)
    # loss = evaluate_generator(generator, generator_dataset)
    # logging.info(f'Evaluation loss: {loss}')
    #
    # # Inference example
    # user_text = "Okay USERNAME I used my fucking phone for 2 minutes and it drains it down 8 fucking percent"
    # inputs = generator.tokenizer.encode(user_text + generator.tokenizer.eos_token, return_tensors='pt')
    # output_text = generator.tokenizer.decode(generator.model.generate(inputs, pad_token_id=generator.tokenizer.eos_token_id, max_new_tokens=50, do_sample=True)[0])
    # logging.info(f'Example: "{output_text}"')

    logging.info('Finished GenBot')


if __name__ == "__main__":
    main()
