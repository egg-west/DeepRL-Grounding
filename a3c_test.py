import numpy as np
import torch
import torch.nn.functional as F
import time
import logging

import env as grounding_env
from models import A3C_LSTM_GA

from torch.autograd import Variable
from constants import *

from tensorboardX import SummaryWriter


def test(rank, args, shared_model):
    writer = SummaryWriter()
    torch.manual_seed(args.seed + rank)

    env = grounding_env.GroundingEnv(args)
    env.game_init()

    model = A3C_LSTM_GA(args)

    if (args.load != "0"):
        print("Loading model ... "+args.load)
        model.load_state_dict(
            torch.load(args.load, map_location=lambda storage, loc: storage))

    model.eval()

    (image, instruction), _, _, _ = env.reset()

    # Print instruction while evaluating and visualizing
    if args.evaluate != 0 and args.visualize == 1:
        print("Instruction: {} ".format(instruction))

    # Getting indices of the words in the instruction
    instruction_idx = []
    for word in instruction.split(" "):
        instruction_idx.append(env.word_to_idx[word])
    instruction_idx = np.array(instruction_idx)

    image = torch.from_numpy(image).float()/255.0
    instruction_idx = torch.from_numpy(instruction_idx).view(1, -1)

    reward_sum = 0
    done = True

    start_time = time.time()

    episode_length = 0
    rewards_list = []
    accuracy_list = []
    episode_length_list = []
    num_episode = 0
    best_reward = 0.0
    test_freq = 50
    print('start testing')
    while True:
        episode_length += 1
        if done:
            if (args.evaluate == 0):
                model.load_state_dict(shared_model.state_dict())

            with torch.no_grad():

                cx = torch.zeros(1, 256)
                hx = torch.zeros(1, 256)
        else:
            with torch.no_grad():
                cx = cx.data
                hx = hx.data
        with torch.no_grad():
            tx = torch.from_numpy(np.array([episode_length])).long()
            img = image.unsqueeze(0)
            instruction_idx = instruction_idx
        value, logit, (hx, cx) = model(
                (img,
                 instruction_idx, (tx, hx, cx)))
        prob = F.softmax(logit, dim=-1)
        action = prob.max(1)[1].data.numpy()

        (image, _), reward, done,  _ = env.step(action[0])

        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        if done:
            print('start recording, reward_sum: ', reward_sum)
            
            num_episode += 1
            writer.add_scalar('test/episode_reward', reward_sum, num_episode)
            rewards_list.append(reward_sum)
            # Print reward while evaluating and visualizing
            if args.evaluate != 0 and args.visualize == 1:
                print("Total reward: {}".format(reward_sum))

            episode_length_list.append(episode_length)
            if reward == CORRECT_OBJECT_REWARD:
                accuracy = 1
            else:
                accuracy = 0
            accuracy_list.append(accuracy)

            if(len(rewards_list) >= test_freq):
                print(" ".join([
                    "Time {},".format(time.strftime("%Hh %Mm %Ss",
                                      time.gmtime(time.time() - start_time))),
                    "Avg Reward {},".format(np.mean(rewards_list)),
                    "Avg Accuracy {},".format(np.mean(accuracy_list)),
                    "Avg Ep length {},".format(np.mean(episode_length_list)),
                    "Best Reward {}".format(best_reward)]))
                writer.add_scalar('test/avg_reward', np.mean(rewards_list), num_episode)
                writer.add_scalar('test/avg_acc', np.mean(accuracy_list), num_episode)
                logging.info(" ".join([
                    "Time {},".format(time.strftime("%Hh %Mm %Ss",
                                      time.gmtime(time.time() - start_time))),
                    "Avg Reward {},".format(np.mean(rewards_list)),
                    "Avg Accuracy {},".format(np.mean(accuracy_list)),
                    "Avg Ep length {},".format(np.mean(episode_length_list)),
                    "Best Reward {}".format(best_reward)]))
                if np.mean(rewards_list) >= best_reward and args.evaluate == 0:
                    torch.save(model.state_dict(),
                               args.dump_location+"model_best")
                    best_reward = np.mean(rewards_list)

                rewards_list = []
                accuracy_list = []
                episode_length_list = []
            reward_sum = 0
            episode_length = 0
            (image, instruction), _, _, _ = env.reset()
            # Print instruction while evaluating and visualizing
            if args.evaluate != 0 and args.visualize == 1:
                print("Instruction: {} ".format(instruction))

            # Getting indices of the words in the instruction
            instruction_idx = []
            for word in instruction.split(" "):
                instruction_idx.append(env.word_to_idx[word])
            instruction_idx = np.array(instruction_idx)
            instruction_idx = torch.from_numpy(instruction_idx).view(1, -1)

        image = torch.from_numpy(image).float()/255.0
