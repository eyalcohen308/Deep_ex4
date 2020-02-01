from torch import optim
from torch import cuda
from utils import *
from model import *

import random


def calculate_accuracy(y_hats, batch_labels):
	good = sum([y_hats[i] == batch_labels[i] for i in range(len(batch_labels))])
	return int(good) / len(batch_labels)


def iterate_model(encoder, atten_model, train_data_loader, encoder_optimizer, atten_optimizer, criterion, epoch):
	percentages_show = 5
	limit_to_print = round(len(train_data_loader) * (percentages_show / 100))
	limit_to_print = max(1, limit_to_print)

	print("\nTrain | Epoch {0}:".format(epoch + 1))

	for index, batch in enumerate(train_data_loader):
		(batch_sen1, batch_sen2), batch_labels = batch
		batch_sen1 = batch_sen1.cuda()
		batch_sen2 = batch_sen2.cuda()
		batch_labels = batch_labels.cuda()

		encoder_optimizer.zero_grad()
		atten_optimizer.zero_grad()

		embed_batch_sen1, embed_batch_sen2 = encoder(batch_sen1, batch_sen2)
		model_output = atten_model(embed_batch_sen1, embed_batch_sen2)

		loss = criterion(model_output, batch_labels)

		loss.backward()
		encoder_optimizer.step()
		atten_optimizer.step()

		# Information printing:
		if index % limit_to_print == 0 and index != 0:
			percentages = index / len(train_data_loader)
			update_progress(percentages)

	print('\r------ Train | Finished epoch {0} ------'.format(epoch + 1))


def update_progress(workdone):
	print("\rProgress: [{0:50s}] {1:.1f}%".format('#' * int(workdone * 50), workdone * 100), end="", flush=True)


def evaluate_accuracy(encoder, atten_model, dev_data_loader, criterion, epoch):
	percentages_show = 5
	limit_to_print = round(len(dev_data_loader) * (percentages_show / 100))
	limit_to_print = max(1, limit_to_print)
	counter = 0
	avg_acc = 0
	avg_loss = 0
	print("\nDev | Epoch {0}:".format(epoch + 1))
	for index, batch in enumerate(dev_data_loader):
		(batch_sen1, batch_sen2), batch_labels = batch
		batch_sen1 = batch_sen1.cuda()
		batch_sen2 = batch_sen2.cuda()
		batch_labels = batch_labels.cuda()
		counter += 1

		embed_batch_sen1, embed_batch_sen2 = encoder(batch_sen1, batch_sen2)
		model_output = atten_model(embed_batch_sen1, embed_batch_sen2)

		loss = criterion(model_output, batch_labels)

		y_hats = torch.argmax(model_output, dim=1)

		current_accuracy = calculate_accuracy(y_hats, batch_labels)
		avg_acc += current_accuracy
		avg_loss += float(loss)

		# Information printing:
		if index % limit_to_print == 0 and index != 0:
			percentages = (index / len(dev_data_loader))
			update_progress(percentages)

	print('\r------ Dev | Finished epoch {0} ------'.format(epoch + 1))

	# Calculating acc and loss on all data set.
	acc = (avg_acc / counter) * 100
	loss = avg_loss / counter

	print("Epoch Accuracy: {0}".format(str(acc)))
	print("Epoch Loss: {0}\n".format(str(loss)))
	return acc, loss


def train(encoder, atten, train_loader, dev_loader, criterion, encoder_optimizer, atten_optimizer, epochs=10):
	dev_acc_list = []
	dev_loss_list = []
	for epoch in range(epochs):
		# train loop
		iterate_model(encoder, atten, train_loader, encoder_optimizer, atten_optimizer, criterion, epoch)

		# calculate performance on dev_data_set
		if CALCULATE_PERFORMANCE_ON_DEV:
			dev_acc, dev_loss = evaluate_accuracy(encoder, atten, dev_loader, criterion, epoch)

			dev_acc_list.append(dev_acc)
			dev_loss_list.append(dev_loss)

	if CALCULATE_PERFORMANCE_ON_DEV:
		print("\n\nTotal Accuracy: {0}".format(str(dev_acc_list)))
		print("\nTotal Loss: {0}\n".format(str(dev_loss_list)))


CALCULATE_PERFORMANCE_ON_DEV = True
USE_GLOVE = False

if __name__ == "__main__":
	cuda.set_device(0)

	train_parser = DataParser('snli_1.0_train.jsonl', with_glove=USE_GLOVE)
	train_data = train_parser.data
	F2I = train_parser.F2I
	L2I = train_parser.L2I
	dev_parser = DataParser('snli_1.0_dev.jsonl', F2I, L2I)
	batch_size = 30
	train_loader = make_loader(train_parser, batch_size)
	random.shuffle(train_loader)
	dev_loader = make_loader(dev_parser, batch_size)
	random.shuffle(dev_loader)

	num_embedding = len(F2I)
	embedding_size = 300
	hidden_size = 400
	lr = 0.05
	model_optimizer = 0.025
	encoder_optimizer = 0.05
	epochs = 10
	label_size = len(L2I)
	criterion = nn.NLLLoss(ignore_index=F2I[PAD])

	encoder = Encoder(num_embedding, embedding_size, hidden_size)
	encoder.cuda()
	atten = Atten(hidden_size, label_size)
	atten.cuda()

	encoder_optimizer = optim.SGD(encoder.parameters(), encoder_optimizer)
	atten_optimizer = optim.SGD(atten.parameters(), model_optimizer)

	# train
	train(encoder, atten, train_loader, dev_loader, criterion, encoder_optimizer, atten_optimizer, epochs)
