import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataloader import *
from evaluation import *
from vdcnn_model import *
torch.cuda.set_device(0)


# train model
def train(batch_size, num_epochs, learning_rate, datapath):
    training_params = {"batch_size": batch_size,
                       "shuffle": True,
                       "num_workers": 0}
    test_params = {"batch_size": batch_size,
                   "shuffle": False,
                   "num_workers": 0}

    validation_params = {"batch_size": batch_size,
                         "shuffle": False,
                         "num_workers": 0}

    training_set = MyDataset(datapath + "train_25m.csv")
    validation_set = MyDataset(datapath + "validation_3m.csv")
    # generators for our training and test sets
    training_generator = DataLoader(training_set, **training_params)
    # test_generator = DataLoader(test_set,**test_params)
    validation_generator = DataLoader(validation_set, **validation_params)

    # our model
    model = Vdcnn(n_classes=2, depth=29, shortcut=True)
    #if torch.cuda.is_available():
    model.cuda()

    # loss function and optimizer
    # using binary cross entropy loss
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # train the model; basically telling on what to train
    model.train()

    num_iter_per_epoch = len(training_generator)

    best_accuracy = 0
    output_file = open(datapath + "logs_vdcnn.txt", "w")
    # with open(datapath + "logs_1.txt", "w") as output_file:
    # training loop
    for epoch in range(num_epochs):

        for iter, batch in enumerate(training_generator):
            # get the inputs
            _, n_true_label = batch

            # wrap them in Variables
            # Variables are specifically tailored to hold values which
            # change during training of a neural network,
            # i.e. the learnable paramaters of our network
            #if torch.cuda.is_available():
            batch = [Variable(record).cuda() for record in batch]
            # else:
            #     batch = [Variable(record) for record in batch]

            # final inputs after wrapping
            t_data, t_true_label = batch

            # forward pass: compute predicted y by passing x to the model
            t_predicted_label = model(t_data)
            # print(t_predicted_label[0])

            # print(t_predicted_label.size(),t_true_label.size())

            # retrieve tensor held by a variable
            n_prob_label = t_predicted_label.cpu().data.numpy()

            # compute loss
            loss = criterion(t_predicted_label, t_true_label)
            # if(iter%1000==0):
            # print("After {} iterations, loss : {}".format(iter+1,loss))

            # Zero gradients, perform a backward pass, and update the weights.
            optimizer.zero_grad()
            # compute gradients
            loss.backward()
            optimizer.step()

            # my useless comment

            training_metrics = get_evaluation(n_true_label, n_prob_label, list_metrics=["accuracy", "loss"])

            print("Training: Iteration: {}/{} Epoch: {}/{} Loss: {} Accuracy: {}".format(iter + 1,
                                                                                         num_iter_per_epoch,
                                                                                         epoch + 1, num_epochs,
                                                                                         training_metrics["loss"],
                                                                                         training_metrics[
                                                                                             "accuracy"]))

        # evaluation of validation data
        model.eval()
        with torch.no_grad():

            validation_true = []
            validation_prob = []

            for batch in validation_generator:
                _, n_true_label = batch

                # setting volatile to true because we are in inference mode
                # we will not be backpropagating here
                # conserving our memory by doing this
                # edit:volatile is deprecated now; using torch.no_grad();see above

                # if torch.cuda.is_available():
                batch = [Variable(record).cuda() for record in batch]
                # else:
                #     batch = [Variable(record) for record in batch]
                # get inputs
                t_data, _ = batch
                # forward pass
                t_predicted_label = model(t_data)
                # using sigmoid to predict the label
                # t_predicted_label = F.sigmoid(t_predicted_label)

                validation_prob.append(t_predicted_label)
                validation_true.extend(n_true_label)

            validation_prob = torch.cat(validation_prob, 0)
            validation_prob = validation_prob.cpu().data.numpy()
            validation_true = np.array(validation_true)

        # back to default:train
        model.train()

        test_metrics = get_evaluation(validation_true, validation_prob,
                                      list_metrics=["accuracy", "loss", "confusion_matrix"])

        output_file.write(
            "Epoch: {}/{} \nTraining loss: {} Training accuracy: {} \nTest loss: {} Test accuracy: {} \nTest confusion matrix: \n{}\n\n".format(
                epoch + 1, num_epochs,
                training_metrics["loss"],
                training_metrics["accuracy"],
                test_metrics["loss"],
                test_metrics["accuracy"],
                test_metrics["confusion_matrix"]))
        print("\tTest:Epoch: {}/{} Loss: {} Accuracy: {}\r".format(epoch + 1, num_epochs, test_metrics["loss"],
                                                                   test_metrics["accuracy"]))

        # acc to the paper; half lr after 3 epochs
        if (num_epochs > 0 and num_epochs % 3 == 0):
            learning_rate = learning_rate / 2

        # saving the model with best accuracy
        if test_metrics["accuracy"] > best_accuracy:
            best_accuracy = test_metrics["accuracy"]
            torch.save(model, datapath + "trained_model_vdcnn")


if __name__ == '__main__':
    # train model and get metrics
    train(512, 10, 0.01, "/home/rachneet/vdcnn/")