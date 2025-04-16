"""
Author: Yu Chen
Email: yu_chen2000@hust.edu.cn
Website: hustyuchen.github.io
"""

import datetime
import numpy as np
import torch
from torch import optim
from initial_utils import SFRA_dataset, selected_training_dataset

from models_util import Normal_decoder, Normal_encoder, initialize_weights
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import os
from sklearn.cluster import KMeans
import pandas as pd
import csv
from active_learning_utils import query_max_uncertainty, query_margin_prob, query_max_entropy, query_margin_kmeans, query_margin_kmeans_2stages, query_margin_kmeans_pure_diversity, random_sampleing
from shutil import copyfile

def intital_setup_and_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device

def training_autoencoder(epochs, batch_size, device, logs_dir, save_model_dir):

    """
    training autoencoder and record best encoder
    Args:
        epochs: training epochs
        batch_size: batch size
        device: use GPU to accelerate computing
        logs_dir: tensorboard dir address
        save_model_dir: model parameters dir address

    Returns:
        None
    """

    # the last paramters can be tunned
    encoder = Normal_encoder(128, 1, 2)
    decoder = Normal_decoder(128, 14, 1, 2)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    initialize_weights(encoder)
    initialize_weights(decoder)

    ## model setting
    print(encoder)
    print(decoder)

    ## training setting
    criteon = nn.MSELoss()
    optimizer_encoder = optim.Adam(encoder.parameters(), lr=2e-4, betas=(0.0, 0.9))
    optimizer_decoder = optim.Adam(decoder.parameters(), lr=2e-4, betas=(0.0, 0.9))

    criteon_test = nn.MSELoss(reduction="sum")

    ## load data
    training_dataset = SFRA_dataset(whether_is_training=True, resize=128)
    testing_dataset = SFRA_dataset(whether_is_training=False, resize=128)
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    testing_dataloader = DataLoader(testing_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    ## Visualization
    wirter = SummaryWriter(logs_dir)
    total_step = 0
    min_test_loss = 99999

    for epoch in range(epochs):

        encoder.train()
        decoder.train()

        for batch_idx, (Gain, Phase, _) in enumerate(training_dataloader):
            Gain = Gain.to(device)
            Phase = Phase.to(device)

            representation_learning = encoder(Gain, Phase)
            Gain_reconstruct, Phase_reconstruct= decoder(representation_learning)
            loss = criteon(Gain, Gain_reconstruct) + criteon(Phase, Phase_reconstruct)

            # backprop
            optimizer_decoder.zero_grad()
            optimizer_encoder.zero_grad()
            loss.backward()
            optimizer_decoder.step()
            optimizer_encoder.step()

            # recode
            wirter.add_scalar("training_loss", loss, global_step=total_step)
            total_step += 1

        print(f"{epoch}/{epochs}, training_loss: {loss.item()}")


        if (epoch % 20) == 0:
            encoder.eval()
            decoder.eval()
            total_loss = 0

            with torch.no_grad():
                for (Gain, Phase, _) in testing_dataloader:
                    Gain = Gain.to(device)
                    Phase = Phase.to(device)
                    representation_learning = encoder(Gain, Phase)
                    Gain_reconstruct, Phase_reconstruct = decoder(representation_learning)
                    loss_test = criteon_test(Gain, Gain_reconstruct) + criteon_test(Phase, Phase_reconstruct)
                    total_loss += loss_test

                test_loss = total_loss / 155
                print("--------------------test------------------------")
                print(f"{epoch}/{epochs}, testing_loss: {test_loss.item()}")
                wirter.add_images("testing_Gain", Gain, epoch)
                wirter.add_images("testing_Phase", Phase, epoch)
                wirter.add_images("testing_Gain_reconstruct", Gain_reconstruct, epoch)
                wirter.add_images("testing_Phase_reconstruct", Phase_reconstruct, epoch)
                wirter.add_scalar("testing_loss", test_loss, epoch)
                print("--------------------end-------------------------")
                # save model
                if test_loss < min_test_loss:
                    min_test_loss = test_loss
                    if not os.path.isdir(save_model_dir):
                        os.mkdir(save_model_dir)
                    torch.save(encoder.state_dict(), os.path.join(save_model_dir,f"encoder_epoch_{epoch}_testing_loss_{test_loss.item()}"))
                    torch.save(decoder.state_dict(), os.path.join(save_model_dir, f"decoder_epoch_{epoch}_testing_loss_{test_loss.item()}"))


def use_encoder_to_select_initial_samples(encoder_state_dict, batch_size, device, training_image_csv_address,
                                          initial_training_csv_address, initial_unselected_csv_address, whether_use_random_sampleing):
    """
    to select batch_size different samples (closest to cluster center) use k-means
    Args:
        encoder_state_dict:  model parameters dict
        batch_size: batch_size different samples (this parameter can be different from batch size of training setting)
        device : use GPU to accelerate computing
        training_image_csv_address : this address is resort

    Returns:
        initial_training_csv_address: csv file load batch_size different samples address
    """

    # load encoder parameters
    encoder = Normal_encoder(128, 1, 2)
    encoder.load_state_dict(torch.load(encoder_state_dict))
    encoder.to(device)
    training_dataset = SFRA_dataset(whether_is_training=True, resize=128)
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # k-means setting

    features = pd.DataFrame()

    encoder.eval()
    for (Gain, Phase, _) in training_dataloader:
        Gain = Gain.to(device)
        Phase = Phase.to(device)
        representation = encoder(Gain, Phase)
        representation = pd.DataFrame(representation.cpu().detach().numpy())
        features = features._append(representation, ignore_index=True)

    # fit kmeans and data select
    if whether_use_random_sampleing == False:
        knn_model = KMeans(n_clusters=batch_size)
        Cluster = knn_model.fit(features)
        distance = Cluster.transform(features)
        selected_sample_list = np.argmin(distance, axis=0)
    else:
        selected_sample_list = random_sampleing(features, batch_size)

    # load sample
    Gain_address_list, Phase_address_lists, label_lists = [], [], []
    with open(training_image_csv_address, mode="r", newline="") as f:
        reader = csv.reader(f)
        reader = list(reader)
        reader_length = len(reader)
        for number in range(reader_length):
            Gain, Phase, label = reader[number]
            Gain_address_list.append(Gain)
            Phase_address_lists.append(Phase)
            label_lists.append(label)

    # this file is to delete selected sample and writer unselected samples' address into csv file
    save_list = list(range(reader_length))
    for selected_sample in selected_sample_list:
        save_list.remove(selected_sample)

    with open(initial_unselected_csv_address, mode="w", newline="") as f:
        writer = csv.writer(f)
        for address_number in save_list:
            writer.writerow([Gain_address_list[address_number], Phase_address_lists[address_number], label_lists[address_number]])

    # this step is to write intial training data address into csv file
    state_count = np.zeros(shape=(1, 7))
    with open(initial_training_csv_address, mode="w", newline="") as f:
        writer = csv.writer(f)
        for address_number in selected_sample_list:
            label_number = int(label_lists[address_number])
            state_count[0, label_number] = state_count[0, label_number] + 1
            writer.writerow([Gain_address_list[address_number], Phase_address_lists[address_number], label_lists[address_number]])

    # this step you should notice that every class that you need one sample at least
    if np.any(label_number == 0):
        print("Some class have no training samples, please check whether encoder has something wrong or encoder is trained bad")
    else:
        print("Initial training dataset is safe")

def training_fault_dignosis_model(whether_training_form_scratch, encoder_state_dict, device, epochs, image_csv_file, logs_dir, active_batch_size, active_learning_step):

    """

    Args:
        whether_training_form_scratch: choose whether fine-tuning encoder of auto-encoder
        encoder_state_dict: encoder parameters
        device: use GPU to
        epochs: training epochs
        image_csv_file: training dataset csv address
        logs_dir: log dir address
        active_learning_step: active learning step

    Returns:
        best_model_parameters_address : return best trained model
        best acc： return best acc
    """

    ## make dir
    if not (os.path.isdir(f"./model_trained_by_active_learning/model_step_{active_learning_step}")):
        os.mkdir(f"./model_trained_by_active_learning/model_step_{active_learning_step}")

    ## training setting
    epochs = epochs
    batch_size = active_batch_size
    learning_rate = 0.005
    loss_function = nn.CrossEntropyLoss()
    total_bk_step = 0
    best_testing_acc_list = [0]
    write = SummaryWriter(logs_dir)

    ## dataset
    selected_dataset = selected_training_dataset(resize=128, training_dataset_address=image_csv_file)
    test_dataset = SFRA_dataset(whether_is_training=False, resize=128)
    selected_dataset_dataloader = DataLoader(selected_dataset, batch_size, shuffle=False, num_workers=0)
    test_dataset_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=0)

    # model setting
    encoder = Normal_encoder(128, 1, 2)
    if not whether_training_form_scratch:
        encoder.load_state_dict(torch.load(encoder_state_dict))
    encoder.add_module("last_linear_layer", nn.Linear(14, 7))
    encoder.to(device)

    print("_______________________Model______________________________")
    print(encoder)
    print("__________________________________________________________")

    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    #optimizer = torch.optim.SGD(encoder.parameters(), lr=learning_rate, momentum=0.1)

    for epoch in range(epochs):
        # training
        encoder.train()
        for (Gain, Phase, label) in selected_dataset_dataloader:
            Gain, Phase, label = Gain.to(device), Phase.to(device), label.to(device)
            output = encoder(Gain, Phase)
            loss = loss_function(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            write.add_scalar("training_loss", loss, total_bk_step)
            total_bk_step += 1

        # testing
        encoder.eval()
        total_accuracy = 0
        with torch.no_grad():
            for (Gain, Phase, label) in test_dataset_dataloader:
                Gain, Phase, label = Gain.to(device), Phase.to(device), label.to(device)
                output = encoder(Gain, Phase)
                accuracy = (output.argmax(1) == label).sum()
                total_accuracy = total_accuracy + accuracy

            accuracy = total_accuracy / 155
            print(f"{epoch}/{epochs}, accuracy:{accuracy}")

            if accuracy > max(best_testing_acc_list):
                torch.save(encoder.state_dict(), f"./model_trained_by_active_learning/model_step_{active_learning_step}/fault_dignosis_model_accuracy_{accuracy}.pt")
                best_model_parameters_address = f"./model_trained_by_active_learning/model_step_{active_learning_step}/fault_dignosis_model_accuracy_{accuracy}.pt"
            best_testing_acc_list.append(accuracy)
            write.add_scalar("test_accuracy", accuracy, epoch)

    write.close()
    best_acc = max(best_testing_acc_list)
    torch.cuda.empty_cache()
    return best_model_parameters_address, best_acc

def using_active_learning_select_samples(best_model_parameters_address, active_learning_strategy, unselected_dataset_csv, training_dataset_csv, batch_size, active_learning_step):
    """

    Args:
        best_model_parameters_address: load best trained encoder
        active_learning_strategy: choose active learning strategy from active learning utils
        unselected_dataset_csv: unselected dataset in last step
        training_dataset_csv: training dataset in last step
        batch_size: active learning's batch size (how much new samples need to be added into training )
        active_learning_step:

    Returns:

    """

    # dataset setting
    unselected_dataset = selected_training_dataset(resize=128, training_dataset_address=unselected_dataset_csv)
    unselected_dataset_loader = DataLoader(unselected_dataset, 32, False, num_workers=0)

    # copy file
    copyfile(training_dataset_csv, f"./training_dataset_address_file/training_dataset_step_{active_learning_step}.csv")

    # model_setting
    encoder = Normal_encoder(128, 1, 2)
    encoder.add_module("last_linear_layer", nn.Linear(14, 7))
    encoder.load_state_dict(torch.load(best_model_parameters_address))
    encoder = encoder.to(device)

    # get representation
    y_preds = pd.DataFrame()
    last_layer_representations = pd.DataFrame()

    for (Gain, Phase, _) in unselected_dataset_loader:
        Gain, Phase = Gain.to(device), Phase.to(device)
        y_pred = encoder(Gain, Phase)
        last_layer_representation = torch.concat([encoder.encoder_cnn_1(Gain), encoder.encoder_cnn_1(Phase)], dim=1)
        y_pred = pd.DataFrame(y_pred.cpu().detach().numpy())
        last_layer_representation = pd.DataFrame(last_layer_representation.cpu().detach().numpy())
        y_preds = y_preds._append(y_pred, ignore_index=True)
        last_layer_representations = last_layer_representations._append(last_layer_representation, ignore_index=True)


    # using active learning strategy
    y_preds = np.array(y_preds)
    last_layer_representations = np.array(last_layer_representations)

    if (active_learning_strategy.__name__ == "query_max_uncertainty") or (active_learning_strategy.__name__ == "query_margin_prob") \
        or (active_learning_strategy.__name__ == "query_max_entropy") or (active_learning_strategy.__name__ =="random_sampleing"):
        new_selected_sample_list = active_learning_strategy(y_preds, batch_size)
    elif active_learning_strategy.__name__ == "query_margin_kmeans":
        new_selected_sample_list = active_learning_strategy(y_preds, last_layer_representations, batch_size)
    elif active_learning_strategy.__name__ == "query_margin_kmeans_pure_diversity":
        new_selected_sample_list = active_learning_strategy(last_layer_representations, batch_size)
    elif active_learning_strategy.__name__ == "query_margin_kmeans_2stages":
        # B : We suggest B choose 3~5
        new_selected_sample_list = active_learning_strategy(y_preds, last_layer_representations, batch_size, 10)


    # revised unselected dataset and build new training_data
    Gain_address_list, Phase_address_lists, label_lists = [], [], []
    with open(unselected_dataset_csv, mode="r", newline="") as f:
        reader = csv.reader(f)
        reader = list(reader)
        reader_length = len(reader)
        for number in range(reader_length):
            Gain, Phase, label = reader[number]
            Gain_address_list.append(Gain)
            Phase_address_lists.append(Phase)
            label_lists.append(label)

    save_list = list(range(reader_length))
    for selected_sample in new_selected_sample_list:
        save_list.remove(selected_sample)

    # write unselected sample dataset csv
    with open(os.path.join("./training_dataset_address_file", f"unselected_dataset_step_{active_learning_step}.csv"), mode="w", newline="") as f:
        writer = csv.writer(f)
        for address_number in save_list:
            writer.writerow([Gain_address_list[address_number], Phase_address_lists[address_number], label_lists[address_number]])

    # write new training dataset csv
    with open(f"./training_dataset_address_file/training_dataset_step_{active_learning_step}.csv", mode="a", newline="") as f:
        writer = csv.writer(f)
        for address_number in new_selected_sample_list:
            writer.writerow([Gain_address_list[address_number], Phase_address_lists[address_number], label_lists[address_number]])



if __name__ == '__main__':
    active_learning_steps = 9  # actually 10
    device = intital_setup_and_seed(3407)
    active_learning_batch_size = 64
    training_epochs = 10
    whether_training_from_scratch = True
    whether_use_initial_random_sample = False

    ## query_max_uncertainty, query_margin_prob, query_max_entropy, query_margin_kmeans, query_margin_kmeans_2stages, query_margin_kmeans_pure_diversity, random_sampleing
    # active_learning_lists = [query_max_uncertainty, query_margin_prob, query_max_entropy, query_margin_kmeans, query_margin_kmeans_2stages, query_margin_kmeans_pure_diversity, random_sampleing]

    active_learning_lists = [query_max_uncertainty, query_margin_prob, query_max_entropy, query_margin_kmeans, query_margin_kmeans_2stages, query_margin_kmeans_pure_diversity, random_sampleing]

    for active_learning in active_learning_lists:

        choose_active_learning_strategy = active_learning
        write = SummaryWriter(f"./active_learning/{choose_active_learning_strategy.__name__}")

        ## frist step is to training antoencoder
        #training_autoencoder(1000, 32, device, "./logs/autoencoders_1002_3", "./autoencoder_model/training_1002_3")

        ## second step is that using trained encoder and kmeans to select batch_size samples as initial_training_csv_address

        use_encoder_to_select_initial_samples("./autoencoder_model/training_1002_3/encoder_epoch_960_testing_loss_908.186279296875", active_learning_batch_size, device, "./training_dataset/images.csv", "./training_dataset_address_file/initial_training_dataset.csv","./training_dataset_address_file/initial_unselected_dataset.csv", whether_use_initial_random_sample)

        ## third step is to training fault dignosis_model

        best_model_parameters_address, best_acc = training_fault_dignosis_model(whether_training_from_scratch,
                                                                                "./autoencoder_model/training_1002_3/encoder_epoch_960_testing_loss_908.186279296875",
                                                                                device,
                                                                                training_epochs,
                                                                                "./training_dataset_address_file/initial_training_dataset.csv",
                                                                                "./logs/training_fault_dignosis_model_step_1", active_learning_batch_size, 1)

        using_active_learning_select_samples(best_model_parameters_address, choose_active_learning_strategy,
                                             "./training_dataset_address_file/initial_unselected_dataset.csv",
                                             "./training_dataset_address_file/initial_training_dataset.csv", active_learning_batch_size, 1)

        write.add_scalar("test_acc", best_acc, active_learning_batch_size)

        for active_learning_step in range(active_learning_steps):

            save_file_number = active_learning_step + 1
            real_active_learning_step = active_learning_step + 2

            starttime = datetime.datetime.now()
            best_model_parameters_address, best_acc = training_fault_dignosis_model(whether_training_from_scratch,
                                                                                    "./autoencoder_model/training_1002_3/encoder_epoch_960_testing_loss_908.186279296875",
                                                                                    device, training_epochs,
                                                                                    f"./training_dataset_address_file/training_dataset_step_{save_file_number}.csv",
                                                                                    f"./logs/training_fault_dignosis_model_step_{real_active_learning_step}",
                                                                                    active_learning_batch_size, real_active_learning_step)
            endtime = datetime.datetime.now()
            write.add_scalar("time", (endtime - starttime).seconds, active_learning_batch_size * real_active_learning_step)

            using_active_learning_select_samples(best_model_parameters_address, choose_active_learning_strategy,
                                                 f"./training_dataset_address_file/unselected_dataset_step_{save_file_number}.csv",
                                                 f"./training_dataset_address_file/training_dataset_step_{save_file_number}.csv",
                                                 active_learning_batch_size, real_active_learning_step)

            write.add_scalar("test_acc", best_acc, active_learning_batch_size * real_active_learning_step)
        write.close()


