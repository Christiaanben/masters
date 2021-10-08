from unittest import TestCase

import torch

from genbot.data import IntentClassificationDataset, IntentPredictionDataset


class DataTestCase(TestCase):

    def setUp(self):
        self.data = [[{'text': 'I need your help.', 'intent': 'help', 'authored': False},
                      {'text': 'How can I help?', 'intent': 'utter_help', 'authored': True},
                      {'text': 'My music is very quiet over bluetooth.',
                       'intent': 'bluetooth+volume',
                       'authored': False},
                      {'text': 'Did you remember to turn up your volume on your bluetooth device?',
                       'intent': 'utter_bluetooth_volume',
                       'authored': True},
                      {'text': 'Ah that worked. Thank you so much.',
                       'intent': 'success',
                       'authored': False},
                      {'text': "Glad that helped. We're always happy to help",
                       'intent': 'utter_pleasure',
                       'authored': True}],
                     [{'text': 'Do you have any specials?', 'intent': 'promo', 'authored': False},
                      {'text': 'Yes, we do. Check out our promo page at URL',
                       'intent': 'utter_promo',
                       'authored': True}]]

    def test_classifcation_dataset_length(self):
        dataset = IntentClassificationDataset(self.data)
        self.assertEqual(len(dataset), 8)

    def test_classifcation_dataset_index(self):
        dataset = IntentClassificationDataset(self.data)

        text, label = dataset[7]
        target_label = dataset.get_label(self.data[1][1].get('intent'))

        self.assertEqual(text, self.data[1][1].get('text'))
        self.assertTrue(torch.equal(label, target_label))

    def test_prediction_dataset_length(self):
        dataset = IntentPredictionDataset(self.data)
        self.assertEqual(len(dataset), 6)

    def test_prediction_dataset_index(self):
        dataset = IntentPredictionDataset(self.data)

        inputs, label = dataset[5]

        target_inputs = dataset._get_label_sequence_tensor([self.data[1][0].get('intent')])
        self.assertTrue(torch.equal(inputs, target_inputs))
        target_label = dataset.get_label(self.data[1][1].get('intent'))
        self.assertTrue(torch.equal(label, target_label))

    def test_dataset_conversations(self):
        dataset = IntentPredictionDataset(self.data)
        self.assertEqual(dataset.conversations, self.data)
