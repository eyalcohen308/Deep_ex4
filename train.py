from torch import optim
from torch import cuda
from utils import *
from model import *

import random


def acc_calc(prediction, labels_batch):
	good = sum([prediction[i] == labels_batch[i] for i in range(len(prediction))])
	return int(good) / len(labels_batch)


def accuracy_n_loss_on_dataset(encoder, net, loader):
	counter = 0
	avg_acc = 0
	avg_loss = 0
	for (sent1_batch, sent2_batch), labels_batch in loader:
		sent1_batch = Variable(sent1_batch.cuda())
		sent2_batch = Variable(sent2_batch.cuda())
		labels_batch = Variable(labels_batch.cuda())
		train_src_linear, train_tgt_linear = encoder(sent1_batch, sent2_batch)
		output = net(train_src_linear, train_tgt_linear)

		prediction = torch.argmax(output, dim=1)

		loss = F.cross_entropy(output, labels_batch)
		# print(loss)

		counter += 1
		avg_acc += acc_calc(prediction, labels_batch)
		avg_loss += float(loss)

	acc = (avg_acc / counter) * 100
	loss = avg_loss / counter
	return acc, loss


def train(encoder, atten, train_loader, dev_loader, lr=0.05, epochs=10):
	display_need = 0
	criterion = nn.NLLLoss(size_average=True)
	encoder_optimizer = optim.SGD(encoder.parameters(), lr)
	atten_optimizer = optim.SGD(atten.parameters(), lr)
	dev_acc_list = []
	dev_loss_list = []

	def handle_data(epoch, acc, loss):
		print(epoch, 'acc', acc)
		print(epoch, 'loss', loss)
		dev_acc_list.append(acc)
		dev_loss_list.append(loss)

	for epoch in range(epochs):
		sentences_seen = 0
		num_of_batch = 0
		acc, loss = accuracy_n_loss_on_dataset(encoder, atten, dev_loader)
		handle_data(epoch, acc, loss)
		for (sent1_batch, sent2_batch), labels_batch in train_loader:
			sent1_batch = Variable(sent1_batch.cuda())
			sent2_batch = Variable(sent2_batch.cuda())
			labels_batch = Variable(labels_batch.cuda())

			display_need += 1
			num_of_batch += 1
			sentences_seen += len(labels_batch)
			if display_need >= 1000:
				display_need -= 1000
				acc, loss = accuracy_n_loss_on_dataset(encoder, atten, train_loader)
				print('epoch', epoch)
				print('sentences seen', sentences_seen)
				print('num of batches', num_of_batch)
				print('acc', acc)
				print('loss', loss)
				print()
			encoder_optimizer.zero_grad()
			atten_optimizer.zero_grad()

			train_src_linear, train_tgt_linear = encoder(sent1_batch, sent2_batch)
			log_prob = atten(train_src_linear, train_tgt_linear)

			loss = criterion(log_prob, labels_batch)
			# print(sentences_seen, float(loss), log_prob[0].detach().numpy())
			loss.backward()
			encoder_optimizer.step()
			atten_optimizer.step()

	plot_graphs(dev_acc_list, dev_loss_list, epochs, "attention model")


# cuda.set_device(0)  # maybe change to 1???
print("cuda is on ", cuda.is_available())

train_parser = DataParser('snli_1.0_train.jsonl')
train_data = train_parser.data
F2I = train_parser.F2I
L2I = train_parser.L2I
dev_parser = DataParser('snli_1.0_dev.jsonl', F2I, L2I)
print(len(train_parser.F2I))
batch_size = 4
train_loader = make_loader(train_parser, batch_size)
random.shuffle(train_loader)
dev_loader = make_loader(dev_parser, batch_size)
random.shuffle(dev_loader)

num_embedding = len(F2I)
embedding_size = 300
hidden_size = 300
label_size = len(L2I)
encoder = Encoder(num_embedding, embedding_size, hidden_size)
encoder.cuda()
atten = Atten(hidden_size, label_size)
atten.cuda()
lr = 0.05
epochs = 100
train(encoder, atten, train_loader, dev_loader, lr, epochs)
