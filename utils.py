import re
import json

import torch
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

PAD = '<pad>'
UNIQUE = '<uuukkk>'
NUMBER = '<num>'


class DataParser:
	def __init__(self, file_name, F2I=None, L2I=None):
		if L2I is None:
			L2I = {}
		if F2I is None:
			F2I = {}
		self.F2I = F2I
		self.L2I = L2I

		self.file_name = file_name
		self.data = []
		self.max_length = 0
		self.parse_the_data()

	def parse_the_data(self):
		vocab = set()
		labels = set()

		json_file = open(self.file_name, 'r')

		for line in json_file:
			d = json.loads(line)
			sent1_words = parse_line(d['sentence1'])
			sent2_words = parse_line(d['sentence2'])
			label = d['gold_label']

			self.data.append(((sent1_words, sent2_words), label))
			#
			# if max(len(sent1_words), len(sent2_words)) > max_length:
			# 	max_length = max(len(sent1_words), len(sent2_words))

			for word in sent1_words:
				vocab.add(word.lower())

			for word in sent2_words:
				vocab.add(word.lower())

			labels.add(label)

		if not self.F2I:
			self.F2I = {f: i for i, f in enumerate(list(sorted(vocab)))}
			self.F2I[UNIQUE] = len(self.F2I)
			self.F2I[PAD] = len(self.F2I)

		if not self.L2I:
			self.L2I = {l: i for i, l in enumerate(list(sorted(labels)))}

		len_1 = len(max(self.data, key=lambda item: len(item[0][0]))[0][0])
		len_2 = len(max(self.data, key=lambda item: len(item[0][1]))[0][1])
		self.max_length = max(len_1, len_2)


def plot_graphs(dev_acc_list, dev_loss_list, iters, name):
	ticks = int(iters / 10)
	if not ticks:
		ticks = 1
	plt.plot(range(iters + 1), dev_acc_list)
	plt.xticks(np.arange(0, iters + 1, step=1))
	plt.yticks(np.arange(0, 110, step=10))
	plt.xlabel('epochs')
	plt.ylabel('accuracy')
	plt.title('{} accuracy'.format(name))
	for i in range(0, len(dev_acc_list), ticks):
		plt.annotate(round(dev_acc_list[i], 1), (i, dev_acc_list[i]))
	plt.show()

	plt.plot(range(iters + 1), dev_loss_list)
	plt.xticks(np.arange(0, iters + 1, step=1))
	plt.yticks(np.arange(0, 4, step=0.5))
	plt.xlabel('epochs')
	plt.ylabel('loss')
	plt.title('{} loss'.format(name))
	for i in range(0, len(dev_loss_list), ticks):
		plt.annotate(round(dev_loss_list[i], 2), (i, dev_loss_list[i]))
	plt.show()


def prepare_list(str_list, max_length, mapper):
	idx_list = []
	if len(str_list) > max_length:
		str_list = str_list[-max_length:]
	for s in str_list:
		if s in mapper:
			idx_list.append(mapper[s])
		else:
			idx_list.append(mapper[UNIQUE])
	while len(idx_list) < max_length:
		idx_list.append(mapper[PAD])
	return idx_list


def parse_line(s):
	punc = ['\"', '\'', '-', '.', '!', '?', ',', '(', ')', '#', '$', '&']
	for p in punc:
		s = s.replace(p, ' {0} '.format(p))
	words_list = s.split()

	for i, word in enumerate(words_list):
		if len(word) == 0:
			del words_list[i]
		elif word[0].isdigit():
			words_list[i] = NUMBER
	# for letter in word:
	#     if letter.isdigit():
	#         words_list[i] = NUMBER
	#         break
	return words_list


def parse_data(file_name):
	data = []
	vocab = set()
	labels = set()
	max_length = 0

	json_file = open(file_name, 'r')

	for line in json_file:
		d = json.loads(line)
		sent1_words = parse_line(d['sentence1'])
		sent2_words = parse_line(d['sentence2'])
		label = d['gold_label']

		data.append(((sent1_words, sent2_words), label))

		if max(len(sent1_words), len(sent2_words)) > max_length:
			max_length = max(len(sent1_words), len(sent2_words))

		for word in sent1_words:
			vocab.add(word.lower())

		for word in sent2_words:
			vocab.add(word.lower())

		labels.add(label)

	L2I = {l: i for i, l in enumerate(list(sorted(labels)))}

	if '' in vocab:
		vocab.remove('')
	F2I = {f: i for i, f in enumerate(list(sorted(vocab)))}
	F2I[UNIQUE] = len(F2I)
	F2I[PAD] = len(F2I)

	return data, F2I, L2I, max_length


def get_sentences_max_len(x):
	(sent1, sent2), _ = x
	return max(len(sent2), len(sent1))


def get_sentence_max_len_from_batch(batch_sentences, sentence_label):
	sentence_choice = 0 if sentence_label == "sentence1" else 1
	return max(len(tup[sentence_choice]) for tup, _ in batch_sentences)


def add_batch(single_batch, batches, F2I, L2I):
	padded_sentence1 = []
	padded_sentence2 = []
	label_matrix = []
	sen1_max_len = get_sentence_max_len_from_batch(single_batch, "sentence1")
	sen2_max_len = get_sentence_max_len_from_batch(single_batch, "sentence2")

	for (sentence1, sentences2), label in single_batch:
		padded_sentence1.append(prepare_list(sentence1, sen1_max_len, F2I))
		padded_sentence2.append(prepare_list(sentences2, sen2_max_len, F2I))
		label_matrix.append(L2I[label])

	batches.append(((padded_sentence1.copy(), padded_sentence2.copy()), label_matrix.copy()))


def tensorize_batches(batches):
	for i, ((sen1_batch, sen2_batch), labels_batch) in enumerate(batches):
		batches[i] = ((torch.LongTensor(sen1_batch), torch.LongTensor(sen2_batch)), torch.LongTensor(labels_batch))


def make_loader(data_parser, batch_size=4):
	data = data_parser.data
	F2I = data_parser.F2I
	L2I = data_parser.L2I
	batches = []

	data = sorted(data, key=get_sentences_max_len)
	iter_num = len(data) // batch_size
	for i in range(iter_num):
		batch = data[batch_size * i: batch_size * (i + 1)]
		add_batch(batch, batches, F2I, L2I)

	last_batch = data[batch_size * iter_num:]
	if last_batch:
		add_batch(last_batch, batches, F2I, L2I)

	tensorize_batches(batches)

	return batches


def analayze_data(data):
	sent1_counter = Counter([len(sent1) for (sent1, sent2), label in data])
	sent2_counter = Counter([len(sent2) for (sent1, sent2), label in data])
	sent1_points = [(i, sent1_counter[i]) for i in range(83)]
	sent2_points = [(i, sent2_counter[i]) for i in range(83)]
	sub_lengths = [abs(len(sent1) - len(sent2)) for (sent1, sent2), label in data]
	print(sum(sub_lengths) / len(sub_lengths))
	print(sum(x > 4 for x in sub_lengths))
	print(sent1_points)
	print(sent2_points)
# plt.plot(sent1_points)
# plt.show()
# plt.plot(sent2_points)
# plt.show()
