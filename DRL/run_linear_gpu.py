import numpy as np
import time


import torch
import torch.nn as nn
import torch.optim as optim

from utils_deep import draw_train_episode, draw_eval_episode

torch.manual_seed(1)

def logging(s, log_path, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(log_path, 'a+') as f_log:
            f_log.write(s + '\n')

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(512, 1, bias = True)
        #self.hidden1 = nn.Linear(64, 8, bias = True)
        #self.hidden2 = nn.Linear(8, 1, bias = True)
        self.affine2 = nn.Linear(1, 1, bias = False)
        torch.nn.init.xavier_uniform_(self.affine1.weight)
        #torch.nn.init.xavier_uniform_(self.affine2.weight)
        #torch.nn.init.xavier_uniform_(self.hidden1.weight)
        #torch.nn.init.xavier_uniform_(self.hidden2.weight)
        self.elu_1 = nn.ELU()
        #self.elu_2 = nn.ELU()
        #self.elu_3 = nn.ELU()
        self.tanh = nn.Tanh()

        self.saved_log_probs = []
        self.rewards = 0
        self.actions = []

    def forward(self, x, y):
        x = self.affine1(x)
        x = self.elu_1(x)
        #x = self.hidden1(x)
        #x = self.elu_2(x)
        #x = self.hidden2(x)
        #x = self.elu_3(x)
        y = self.affine2(y)
        action = self.tanh(x + y)
        return action

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print (device)
global policy
policy = Policy()
policy.to(device)

def train_eval(config):
    #change the optimizer
    #add in dropout
    #use new data from Iris
    optimizer = optim.SGD(policy.parameters(), lr= config.init_lr)
    # eps = np.finfo(np.float32).eps.item()
    rewards_over_time = []

    NUM_OF_EVAL_DATA = config.num_of_eval
    
    _time = time.strftime("%Y%m%d-%H%M%S")
    PATH = './linear/best_model_'+ config.currency +'_week' + str(config.week_num) + '_' + _time + '.pth'
    log_path = './linear/log_'+ config.currency +'_week' + str(config.week_num) + '_' + _time + '.txt'

    best_accumulative_return = -1000

    for epoch in range(config.num_of_epoch):
        for i_episode in range(config.num_of_episode):
            ask = torch.zeros((1, 1)).to(device)
            bid = torch.zeros((1, 1)).to(device)
            previous_action = torch.tensor([0.0]).to(device)
            while ask.size()[0] <= config.timespan and bid.size()[0]<= config.timespan:
                target_bid, target_ask, feature_span = draw_train_episode(config.week_num, config.lag, config.currency, config.min_history)
                bid, ask, features = target_bid*1e3, target_ask*1e3, feature_span
            for t in range(config.timespan):  # Don't infinite loop while learning
                state = feature_span[t]
                save_action = policy(state.float(),0.1*previous_action).to(device)

                if t == config.timespan-1:
                    save_action = 0

                action = save_action - previous_action
                price = 0
                if action > 0:
                    price = ask[t]
                elif action < 0:
                    price = bid[t]
                reward = torch.sum(torch.tensor(-1.).float() * action * price).to(device)

                policy.rewards += reward

                previous_action = save_action

            optimizer.zero_grad()
            loss = - policy.rewards / config.timespan
            loss.backward(retain_graph=True)
            optimizer.step()
            if i_episode %  10 ==  0:
                to_log = 'Epoch: {} Episode:{} The loss of training is {}'.format(epoch, i_episode, loss.item())
                logging(to_log, log_path)
            policy.rewards = 0

        # eval after running 1000 episodes
        policy.eval()

        with torch.no_grad():
            accumulative_reward_test = 0
            for j in range(NUM_OF_EVAL_DATA):
                current_reward = 0
                ask = np.zeros((1, 1))
                bid = np.zeros((1, 1))
                previous_action = torch.tensor([0.0]).to(device)
                while ask.shape[0] <= config.timespan and bid.shape[0]<=config.timespan:
                    target_bid, target_ask, feature_span = draw_eval_episode(config.week_num, config.lag, config.currency,
                                                                             config.min_history, j, config.offset)
                    bid, ask, features = target_bid*1e3, target_ask*1e3, feature_span
                for t in range(config.timespan):  # Don't infinite loop while learning
                    state = feature_span[t]
                    save_action = policy(state.float(),0.1*previous_action)

                    if t == config.timespan-1:
                        save_action = 0
                    action = save_action - previous_action

                    price = 0
                    if action > 0:
                        price = ask[t]
                    elif action < 0:
                        price = bid[t]
                    reward = torch.sum(torch.tensor(-1.).float() * action * price).to(device)
                    accumulative_reward_test += reward
                    current_reward  += reward
                    previous_action = save_action
            to_log = "Evaluating on {} datapoint and return is {}".format(NUM_OF_EVAL_DATA, accumulative_reward_test)
            logging(to_log, log_path)
            rewards_over_time.append(accumulative_reward_test)

            if (accumulative_reward_test * 1.0 / NUM_OF_EVAL_DATA > best_accumulative_return):
                torch.save(policy.state_dict(), PATH)
                best_accumulative_return = accumulative_reward_test * 1.0 / NUM_OF_EVAL_DATA
        logging("=======================================================", log_path)
        policy.train()

    with open(config.reward_file, 'w') as filehandle:
        for listitem in rewards_over_time:
            filehandle.write('%s\n' % listitem)


if __name__ == '__main__':
    train_eval(config)
