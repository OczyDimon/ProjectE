# import warnings
#
# # from datasets import load_dataset
# #
# # ds = load_dataset("DrishtiSharma/Anime-Face-Dataset", split="train")
#
# warnings.filterwarnings("ignore")
#
# model = Sequential([
#         # input is Z, going into a convolution
#         layers.Conv2DTranspose(z_dim, 64 * 8, 4, 1, 0),
#         layers.BatchNorm2d(64 * 8),
#         layers.ReLU(True),
#         # state size. ``(ngf*8) x 4 x 4``
#         layers.Conv2DTranspose(64 * 8, 64 * 4, 4, 2, 1),
#         layers.BatchNorm2d(64 * 4),
#         layers.ReLU(True),
#         # state size. ``(ngf*4) x 8 x 8``
#         layers.Conv2DTranspose(64 * 4, 64 * 2, 4, 2, 1),
#         layers.BatchNorm2d(64 * 2),
#         layers.ReLU(True),
#         # state size. ``(ngf*2) x 16 x 16``
#         layers.Conv2DTranspose(64 * 2, 64, 4, 2, 1),
#         layers.BatchNorm2d(64),
#         layers.ReLU(True),
#         # state size. ``(ngf) x 32 x 32``
#         layers.Conv2DTranspose(64, 3, 4, 2, 1),
#         layers.Tanh()
#     ])
#
# model = Sequential([
#         # input is ``(nc) x 64 x 64``
#         layers.Conv2d(3, 64, 4, 2, 1),
#         layers.LeakyReLU(0.2, inplace=True),
#         # state size. ``(ndf) x 32 x 32``
#         layers.Conv2d(64, 64 * 2, 4, 2, 1),
#         layers.BatchNorm2d(64 * 2),
#         layers.LeakyReLU(0.2, inplace=True),
#         # state size. ``(ndf*2) x 16 x 16``
#         layers.Conv2d(64 * 2, 64 * 4, 4, 2, 1),
#         layers.BatchNorm2d(64 * 4),
#         layers.LeakyReLU(0.2, inplace=True),
#         # state size. ``(ndf*4) x 8 x 8``
#         layers.Conv2d(64 * 4, 64 * 8, 4, 2, 1),
#         layers.BatchNorm2d(64 * 8),
#         layers.LeakyReLU(0.2, inplace=True),
#         # state size. ``(ndf*8) x 4 x 4``
#         layers.Conv2d(64 * 8, 1, 4, 1, 0),
#         layers.Sigmoid()
#     ])
