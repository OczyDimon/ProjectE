# from math import factorial
#
# from dataset_generator import max_size, number_of_samples
#
# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
#
# import keras
# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense, Flatten, Input
# from keras.optimizers import Adam
#
#
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# model = Sequential()
#
# # model.add(Flatten(input_shape=(max_size, max_size)))
# # model.add(Dense(1000))
# # model.add(Dense(100))
# # model.add(Dense(1))
# # model.add(Flatten(input_shape=(max_size, max_size)))
# # model.add(Dense(64))
# model.add(Input(shape=(max_size, max_size)))
# model.add(Flatten(input_shape=(max_size, max_size)))
# model.add(Dense(factorial(8)))
# model.add(Dense(1, activation='linear'))
#
# adam = Adam()
# model.compile(optimizer=adam,
#               loss='mean_squared_error',
#               metrics=['accuracy'],
#               )
#
# model.fit(x_train, y_train, epochs=100)
#
# loss, accuracy = model.evaluate(x_test, y_test)
# print('Accuracy:', accuracy)


"""
Import necessary libraries to create a generative adversarial network
The code is mainly developed using the PyTorch library
"""
#
# import time
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision.transforms import transforms
# from model import discriminator, generator
# import numpy as np
# import matplotlib.pyplot as plt
# from datasets import load_dataset
#
# ds = load_dataset("DrishtiSharma/Anime-Face-Dataset", split="train")
#
#
# """
# Determine if any GPUs are available
# """
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#
# """
# Hyperparameter settings
# """
# epochs = 150
# lr = 2e-4
# batch_size = 64
# loss = nn.BCELoss()
#
# # Model
# G = generator().to(device)
# D = discriminator().to(device)
#
# G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
# D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
#
#
# """
# Image transformation and dataloader creation
# Note that we are training generation and not classification, and hence
# only the train_loader is loaded
# """
#
# # Transform
# transform = transforms.Compose([transforms.ToTensor(),
#                                 transforms.Normalize((0.5,), (0.5,))])
#
# # Load data
# train_set = ds
# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
#
#
# """
# Network training procedure
# Every step both the loss for disciminator and generator is updated
# Discriminator aims to classify reals and fakes
# Generator aims to generate images as realistic as possible
# """
# for epoch in range(epochs):
#     for idx, (imgs, _) in enumerate(train_loader):
#         idx += 1
#
#         # Training the discriminator
#         # Real inputs are actual images of the MNIST dataset
#         # Fake inputs are from the generator
#         # Real inputs should be classified as 1 and fake as 0
#         real_inputs = imgs.to(device)
#         real_outputs = D(real_inputs)
#         real_label = torch.ones(real_inputs.shape[0], 1).to(device)
#
#         noise = (torch.rand(real_inputs.shape[0], 128) - 0.5) / 0.5
#         noise = noise.to(device)
#         fake_inputs = G(noise)
#         fake_outputs = D(fake_inputs)
#         fake_label = torch.zeros(fake_inputs.shape[0], 1).to(device)
#
#         outputs = torch.cat((real_outputs, fake_outputs), 0)
#         targets = torch.cat((real_label, fake_label), 0)
#
#         D_loss = loss(outputs, targets)
#         D_optimizer.zero_grad()
#         D_loss.backward()
#         D_optimizer.step()
#
#         # Training the generator
#         # For generator, goal is to make the discriminator believe everything is 1
#         noise = (torch.rand(real_inputs.shape[0], 128)-0.5)/0.5
#         noise = noise.to(device)
#
#         fake_inputs = G(noise)
#         fake_outputs = D(fake_inputs)
#         fake_targets = torch.ones([fake_inputs.shape[0], 1]).to(device)
#         G_loss = loss(fake_outputs, fake_targets)
#         G_optimizer.zero_grad()
#         G_loss.backward()
#         G_optimizer.step()
#
#         if idx % 100 == 0 or idx == len(train_loader):
#             print('Epoch {} Iteration {}: discriminator_loss {:.3f}'
#                   ' generator_loss {:.3f}'.format(epoch, idx, D_loss.item(), G_loss.item()))
#
#     if (epoch+1) % 10 == 0:
#         torch.save(G, 'Generator_epoch_{}.pth'.format(epoch))
#         print('Model saved.')
#
# import cv2
# import numpy as np
#
# img = cv2.imread("neural network/matrixes_noised/page_1_matrix_3_593.png")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# thresh = 100
#
# # get threshold image
# ret, thresh_img = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
#
# # find contours without approx
# contours, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#
# max_ = 0
# sel_countour = None
# for countour in contours:
#     if countour.shape[0] > max_:
#         sel_countour = countour
#         max_ = countour.shape[0]
#
# # calc arclentgh
# arclen = cv2.arcLength(sel_countour, True)
#
# # do approx
# eps = 0.0005
# epsilon = arclen * eps
# approx = cv2.approxPolyDP(sel_countour, epsilon, True)
#
# # draw the result
# canvas = img.copy()
# for pt in approx:
#     cv2.circle(canvas, (pt[0][0], pt[0][1]), 7, (0, 255, 0), -1)
#
# cv2.drawContours(canvas, [approx], -1, (0, 0, 255), 2, cv2.LINE_AA)
#
# img_contours = np.uint8(np.zeros((img.shape[0], img.shape[1])))
# cv2.drawContours(img_contours, [approx], -1, (255, 255, 255), 1)
#
#
# cv2.imshow('origin', canvas)  # выводим итоговое изображение в окно
# cv2.imshow('res', img_contours)  # выводим итоговое изображение в окно
#
# cv2.waitKey()
# cv2.destroyAllWindows()


