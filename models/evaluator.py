import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from termcolor import colored, cprint

class LogisticRegression_custom(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression_custom, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.reset_parameters()

    def forward(self, x):
        return self.linear(x)

    def reset_parameters(self):
        self.linear.reset_parameters()
        
def linear_classifier_custom(config, pretrained_representations, data, verbose=False):
    accuracy = 0
    for i in range(config.runs):
        # X_train, X_test, y_train, y_test = train_test_split(pretrained_representations, data.y[data.train_mask], test_size=0.2, random_state=42, stratify=data.y)
        X_train = pretrained_representations[data.train_mask]
        y_train = data.y[data.train_mask]
        X_test = pretrained_representations[data.test_mask]
        y_test = data.y[data.test_mask]

        classifier = LogisticRegression_custom(pretrained_representations.shape[1], config.num_classes)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=config.eval_lr)

        # Training the classifier
        num_epochs = config.eval_epochs

        for epoch in range(num_epochs):
            classifier.train()

            # Forward pass
            outputs = classifier(X_train)
            loss = criterion(outputs, y_train)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss every 10 epochs
            # if (epoch + 1) % 50 == 0:
            #     print(f'Epoch [{epoch + 1}/{num_epochs}], enc_classifier Loss: {loss.item():.4f}')

        classifier.eval()
        with torch.no_grad():
            predicted = classifier(X_test)
            _, pred = torch.max(predicted, 1)
            accuracy += accuracy_score(y_test.numpy(), pred.numpy())
            # print(f'Test Accuracy: {accuracy:.4f}')
    
    avg_acc = round(accuracy / config.runs, 3)
    
    if verbose:
        avg_acc_c = colored(avg_acc, "green", attrs=["bold"])
        print("linear classifier accuracy (MP-JEPA): ", avg_acc_c)
    
    return avg_acc

def linear_classifier(config, pretrained_representations, data, base=False, verbose=False):
    accuracy = 0
    for i in range(config.eval_runs):
        X_train = pretrained_representations[data.train_mask]
        y_train = data.y[data.train_mask]
        X_test = pretrained_representations[data.test_mask]
        y_test = data.y[data.test_mask]

        classifier = LogisticRegression(random_state=42, max_iter=10000)
        
        classifier.fit(X_train.detach().numpy(), y_train.detach().numpy())
        predictions = classifier.predict(X_test.detach().numpy())

        accuracy += accuracy_score(y_test.detach().numpy(), predictions)
    
    avg_acc = round(accuracy / config.runs, 3)
    if verbose:
        if base:
            avg_acc_c = colored(avg_acc, "green", attrs=["bold"])
            print("linear classifier accuracy (Base): ", avg_acc_c)
        else:
            avg_acc_c = colored(avg_acc, "green", attrs=["bold"])
            print("linear classifier accuracy (MP-JEPA): ", avg_acc_c)
    return avg_acc