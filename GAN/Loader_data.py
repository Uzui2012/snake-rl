import pickle
import numpy as np
from PIL.ImageFile import ImageFile
from torch.nn import BCELoss, CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn.functional import mse_loss
from torch.optim import SGD
from torch.optim.adam import Adam
from torch.utils.data import Dataset
from PIL import Image
import os.path as osp
import os
from torchvision.transforms import functional as TF
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import traceback
#import discriminator
# from GAN.discriminator import DiscriminatorSmall
from model import UNet
from torch.utils.data import DataLoader

from reward_model import reward_model
from testing_architectures import GeneratorSmall, DiscriminatorSmall

ImageFile.LOAD_TRUNCATED_IMAGES = True

proj_path = "C:\\Users\\killi\\Documents\\Repositories\\snake-rl\\"

class ImageDataset(Dataset):

    def __init__(self, dataset, transform=None, num_images=1, val=False):
        self.dataset = dataset
        self.transform = transform
        self.val = val

    def __len__(self):
        return len(self.dataset)-1 #The reason index out of range??


    def __getitem__(self, index):
        try :
            if not self.val:
                #print(index)
                #print(self.dataset[index])
                path = self.dataset[index]  # Remember that S_images has 1 image more than Sa_images because index ffor Sa is index-1
                path_output = self.dataset[index].replace("S_images", "Sa_images")
                #print(str(index) + " Worked\n")
                pickled_arr, _ = pickle.load(open(path, "rb"))
                pickled_arr_output = pickle.load(open(path_output, "rb"))
                noise = torch.randn(1, 84, 84)
                noise = noise
                # if self.transform is not None:
                #     img = self.transform(img)
                # for experimental self.transform(pickled_arr_output.to(torch.float32))
                return  pickled_arr, pickled_arr_output  # was pickled_arr, pickled_arr_output
        except Exception:
            
            print(str(index) + " scuffed\n")
            traceback.print_exc()
            exit(555)

class DeblurDataset(object):
    def __init__(self, data_path):
        self.data_val = []
        self.data_train = []
        self.data_path = data_path

    def get_paths(self):

        for folder in os.listdir(osp.join(self.data_path)):
            for fname in os.listdir(self.data_path + "\\" + folder):
                if folder == "Sa_images":

                    self.data_val.append(osp.join(self.data_path, folder, fname))
                elif folder == "S_images":
                    self.data_train.append(osp.join(self.data_path, folder, fname))

        return self


class RewardDataset(Dataset):

    def __init__(self, dataset, transform=None, num_images=1, val=False):
        self.dataset = dataset
        self.transform = transform
        self.val = val

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if not self.val:
            path = self.dataset[
                index]
            path_future = self.dataset[
                index].replace("now", "future")
            (state, reward) = pickle.load(open(path, "rb"))
            # plt.imshow(state, cmap='gray')
            # plt.show()

            future = pickle.load(open(path_future, "rb"))

            # plt.imshow(future, cmap='gray')
            # plt.show()
            state_future = torch.cat([state.unsqueeze(0), future.unsqueeze(0)], 0)
            return state_future, reward


class RewardPathsDataset(object):
    def __init__(self, data_path):
        self.data_val = []
        self.data_train = []
        self.data_path = data_path

    def get_paths(self):

        for folder in os.listdir(osp.join(self.data_path)):
            for fname in os.listdir(self.data_path + "\\" + folder):
                self.data_train.append(osp.join(self.data_path, folder, fname))

        return self


def train_reward_model(path_to_images):
    from torchvision import transforms
    train_transforms_list = [transforms.ToTensor(),
                             # transforms.Normalize(mean, std)
                             ]
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    train_transforms_list = [transforms.ToTensor(),
                             transforms.ToPILImage()]
    train_transforms = transforms.Compose(train_transforms_list)
    data_train = RewardDataset(balance_files(path_to_images), transform=train_transforms)
    data_train_loader = DataLoader(data_train, batch_size=64, shuffle=True, num_workers=8)

    model = reward_model(5).cuda()
    #model.load_state_dict(torch.load(proj_path + "GAN_models\\reward_predictor.pt"))
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

    plot_loss = []
    plot_accuracy = []

    for epoch in range(16):
        model.train()
        running_loss = 0.0
        counter = 0
        correct = 0
        for i, img in enumerate(data_train_loader):
            counter+=1
            state, actual_reward = img
            optimizer.zero_grad()

            actual_reward = actual_reward.float().cuda()
            state = state.float().cuda()
            models_reward = model(state)
            actual_reward = torch.argmax(actual_reward, dim=1)
            loss_reward = CrossEntropyLoss()(models_reward, actual_reward.long())

            running_loss += loss_reward.item()

            plot_loss.append(loss_reward.item())
            
            loss_reward.backward()
            if torch.argmax(models_reward[0]).item() == actual_reward[0].item():
                correct+=1
            print("predicted reward {} actual reward {}".format(torch.argmax(models_reward[0]).item(),
                                                                actual_reward[0].item()))
            print(f"loss image {running_loss / (i + 1)}")
            print(f"accuracy {correct/counter * 100}")
            plot_accuracy.append(correct/counter * 100)
            optimizer.step()
    torch.save(model.state_dict(), proj_path + f"IBP_GAN_Folder_2\\IBP_GAN_Reward_Predictor\\5x5_reward_predictor.pt")
    plt.clf()
    plt.plot(plot_loss)
    plt.savefig(proj_path + f"IBP_GAN_Folder_2\\IBP_GAN_Reward_Predictor\\5x5_reward_predictor_loss")
    plt.clf()
    plt.plot(plot_accuracy)
    plt.savefig(proj_path + f"IBP_GAN_Folder_2\\IBP_GAN_Reward_Predictor\\5x5_reward_predictor_accuracy")


def balance_files(path):
    data_path = path + "\\train_reward"
    counter_1 = 0
    data_path_1 = []
    counter_2 = 0
    data_path_2 = []
    counter_3 = 0
    data_path_3 = []
    for fname in os.listdir(data_path + "\\"):
        for plik in os.listdir(data_path + "\\" + fname):
            if fname == "now":
                path_to_file = data_path + "\\" + fname + "\\" + plik

                state, reward = pickle.load(open(path_to_file, "rb"))

                if torch.all(torch.eq(reward, torch.Tensor([1, 0, 0]))):
                    counter_1 += 1
                    data_path_1.append(path_to_file)
                elif torch.all(torch.eq(reward, torch.Tensor([0, 1, 0]))):
                    counter_2 += 1
                    data_path_2.append(path_to_file)
                elif torch.all(torch.eq(reward, torch.Tensor([0, 0, 1]))):
                    counter_3 += 1
                    data_path_3.append(path_to_file)
            print(f"{counter_1} {counter_2} {counter_3}")
    print(f"{counter_1} {counter_2} {counter_3}")
    lowest_value = min(len(data_path_1), len(data_path_2), len(data_path_3))
    balanced_list = data_path_1[:lowest_value] + data_path_2[:lowest_value] + data_path_3[:lowest_value]
    data_path_1 = data_path_1[:lowest_value]
    data_path_2 = data_path_2[:lowest_value]
    data_path_3 = data_path_3[:lowest_value]
    return data_path_1 + data_path_2 + data_path_3


def train_gan():
    for GAN_num in range(1, 10):
        from torchvision import transforms
        train_transforms_list = [transforms.ToTensor(),
                                # transforms.Normalize(mean, std)
                                ]
        from torch.utils.data import DataLoader
        import matplotlib.pyplot as plt
        counter = 0
        train_transforms_list = [transforms.ToTensor(),
                                transforms.ToPILImage()]
        train_transforms = transforms.Compose(train_transforms_list)
        print(len(DeblurDataset(data_path).get_paths().data_train))
        data_train = ImageDataset(DeblurDataset(data_path).get_paths().data_train, transform=train_transforms)
        data_train_loader = DataLoader(data_train, batch_size=32,shuffle=True,num_workers=4) #comeback shuffle=True
        print(len(data_train_loader.dataset))
        model = UNet(5, 1).cuda()
        #model.load_state_dict(torch.load("C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\new_models\\GAN11_new.pt"))
        optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
        discriminator = DiscriminatorSmall(2).cuda()
        #discriminator.load_state_dict(
        #    torch.load("C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\GAN_models\\DISC1235678RIMINATOR3.pt"))
        optimizer_reward = Adam(model.parameters(), lr=3e-4)
        
        #plot_reward = []
        plot_reward = []
        plot = []
        optimizer_discrimnator = SGD(discriminator.parameters(), lr=0.01, momentum=0.9)
        l1_loss = torch.nn.L1Loss().cuda()

        gan_loss = torch.nn.BCELoss().cuda()

        # while x <= 1000:
        #     alpha = x
        generator_amplifier = 3
        discriminator_deamplifier = 15
        
        for epoch in range(25):
            counter = 0
            print(epoch)
            model.train()
            running_loss = 0.0
            starting_gan = 0.1
            for i, img in enumerate(data_train_loader):
                state_action, resultant_state = img
                optimizer.zero_grad()

                resultant_state = resultant_state.float().cuda()
                state_action = state_action.float().cuda()
                gen_guess = model(state_action)

                loss = l1_loss(gen_guess, resultant_state)
                generator = discriminator(gen_guess)
                loss_gan = gan_loss(generator, torch.ones_like(generator))
                running_loss += loss.item()

                plot.append(loss.item())
                dis_mse_loss = loss*generator_amplifier + loss_gan/discriminator_deamplifier
                dis_mse_loss.backward()

                # print(f"loss image {running_loss / (i + 1)} for alpha {alpha}")
                if i % 100 == 0:
                    print(f"combined Loss : {dis_mse_loss} ") #with starting gan being {starting_gan}")
                optimizer.step()

                optimizer_discrimnator.zero_grad()
                disc_true = discriminator(resultant_state)
                disc_fake = discriminator(gen_guess.detach())
                disc_true_loss = gan_loss(disc_true, torch.ones_like(disc_true))
                disc_fake_loss = gan_loss(disc_fake, torch.zeros_like(disc_fake))

                discriminator_loss = disc_true_loss + disc_fake_loss
                discriminator_loss.backward()
                optimizer_discrimnator.step()
                # if i == len(data_train_loader)-1:
                #     last_deblurs = img_deblur
            
            print(f"finished epoch {epoch}")
            # plt.plot(plot)
            # plt.show()
            #if epoch % 2 == 0:
            #    torch.save(discriminator.state_dict(),
            #            proj_path + f"new_models\\discriminator13_5x5_{generator_amplifier}_{discriminator_deamplifier}_new_1.pt")
            #    torch.save(model.state_dict(),
            #           proj_path + f"new_models\\GAN13_5x5_{generator_amplifier}_{discriminator_deamplifier}_new_1.pt")
            #plt.plot(plot_reward)
            #plt.show()

            input_image = state_action[0][0].detach().cpu().numpy().squeeze()
            predicted_output_img = gen_guess[0].detach().cpu().numpy().squeeze()
            actual_output = resultant_state[0].detach().cpu().numpy().squeeze()
            plt.imshow(input_image, cmap='gray', vmin=0, vmax=1)
            save_plot_and_dump_pickle(counter, input_image,"input", epoch, GAN_num)
            #plt.show()
            counter += 1
            plt.imshow(predicted_output_img, cmap='gray', vmin=0, vmax=1)
            plt.savefig(proj_path + f'IBP_GAN_Folder_2\\IBP_GAN_tests\\5x5_{counter}_gan_response_GAN-{GAN_num}_epoch-{epoch}_Unique', bbox_inches='tight')
            #.show()
            counter += 1
            plt.imshow(actual_output, cmap='gray', vmin=0, vmax=1)
            plt.savefig(proj_path + f'IBP_GAN_Folder_2\\IBP_GAN_tests\\5x5_{counter}_ground_truth_GAN-{GAN_num}_epoch-{epoch}_Unique',
                            bbox_inches='tight')
            #plt.show()#predicted_output_img = img_deblur[0].detach().cpu().numpy().squeeze()
            
            torch.save(discriminator.state_dict(),
                    proj_path + f"IBP_GAN_Folder_2\\IBP_GAN_Models\\discriminator_Unique_5x5_{generator_amplifier}_{discriminator_deamplifier}_num{GAN_num}_epoch{epoch}.pt")
            torch.save(model.state_dict(),
                    proj_path + f"IBP_GAN_Folder_2\\IBP_GAN_Models\\GAN_Unique_5x5_{generator_amplifier}_{discriminator_deamplifier}_num{GAN_num}_epoch{epoch}.pt")
            plt.clf()
            plt.plot(plot)
            plt.savefig(proj_path + f'IBP_GAN_Folder_2\\IBP_GAN_loss_images\\5x5_loss_GAN-{GAN_num}_epoch-{epoch}_Unique')
            #plt.show()
            # x*=10
            
        


def save_plot_and_dump_pickle(counter, input_image, source, epoch, GAN_num):
    plt.savefig(proj_path + f'IBP_GAN_Folder_2\\IBP_GAN_tests\\5x5_{counter}_input_GAN-{GAN_num}_epoch-{epoch}_Unique', bbox_inches='tight')
    with open(proj_path + f"IBP_GAN_Folder_2\\IBP_GAN_tests\\5x5_{counter}_{source}_GAN-{GAN_num}_epoch-{epoch}_Unique.pickle", 'wb') as handle:
        pickle.dump(input_image, handle, protocol=pickle.HIGHEST_PROTOCOL)


def validate_prepare_data(data_path):
    pickled_arr = pickle.load(
        open(data_path + "\\Sa_images\\5x5_state_s_77017.pickle", "rb"))
    pickled_arr_output = pickle.load(
        open(data_path + "\\S_images\\5x5_state_s_77017.pickle", "rb"))

    return (pickled_arr, pickled_arr_output)


def experimental_train():
    train_transforms_list = [transforms.ToPILImage(),
                             transforms.Resize((20, 20)),
                             transforms.ToTensor()]
    train_transforms = transforms.Compose(train_transforms_list)

    data_train = ImageDataset(DeblurDataset(data_path).get_paths().data_train, transform=train_transforms)
    data_train_loader = DataLoader(data_train, batch_size=64, shuffle=True, num_workers=8)

    discriminator = DiscriminatorSmall(32).cuda()
    model = GeneratorSmall(32).cuda()

    optim_discriminator = torch.optim.Adam(discriminator.parameters(), lr=3e-4, betas=(0.5, 0.999))
    optim_generator = torch.optim.Adam(model.parameters(), lr=3e-4, betas=(0.5, 0.999))
    crit_discriminator = torch.nn.BCELoss().cuda()
    crit_generator = torch.nn.BCELoss().cuda()
    for _ in range(8):
        model.train()
        discriminator.train()
        for i, img in enumerate(data_train_loader):
            img_blur, img_sharp = img

            img_sharp = img_sharp.float().cuda()
            img_blur = img_blur.float().cuda()

            # Generate noise
            noise = torch.randn(64, 100, 1, 1).cuda()
            fake_imgs = model(noise)

            optim_discriminator.zero_grad()
            # train with real
            pred_true = discriminator(img_sharp)
            loss_disc_true = crit_discriminator(pred_true, torch.ones_like(pred_true))

            # train with fake
            pred_fake = discriminator(fake_imgs.detach())
            loss_disc_fake = crit_discriminator(pred_fake, torch.zeros_like(pred_fake))

            loss_disc = loss_disc_true + loss_disc_fake
            loss_disc.backward()

            optim_discriminator.step()
            print(loss_disc.item())
            # Generator
            optim_generator.zero_grad()

            pred_fake_gen = discriminator(fake_imgs)
            loss_gen = crit_generator(pred_fake_gen, torch.ones_like(pred_fake_gen))
            loss_gen.backward()

            optim_generator.step()

            # second iteration
            optim_generator.zero_grad()
            noise = torch.randn(64, 100, 1, 1).cuda()
            fake_imgs = model(noise)
            pred_fake_gen = discriminator(fake_imgs)
            loss_gen = crit_generator(pred_fake_gen, torch.ones_like(pred_fake_gen))
            loss_gen.backward()

            optim_generator.step()

            # second iteration
            optim_generator.zero_grad()
            noise = torch.randn(64, 100, 1, 1).cuda()
            fake_imgs = model(noise)
            pred_fake_gen = discriminator(fake_imgs)
            loss_gen = crit_generator(pred_fake_gen, torch.ones_like(pred_fake_gen))
            loss_gen.backward()

            optim_generator.step()

            # second iteration
            optim_generator.zero_grad()
            noise = torch.randn(64, 100, 1, 1).cuda()
            fake_imgs = model(noise)
            pred_fake_gen = discriminator(fake_imgs)
            loss_gen = crit_generator(pred_fake_gen, torch.ones_like(pred_fake_gen))
            loss_gen.backward()

            optim_generator.step()

        plt.imshow(fake_imgs[0].detach().cpu().numpy().squeeze(), cmap='gray', vmax=1, vmin=0)
        plt.show()
        plt.imshow(fake_imgs[1].detach().cpu().numpy().squeeze(), cmap='gray', vmax=1, vmin=0)
        plt.show()
        plt.imshow(fake_imgs[2].detach().cpu().numpy().squeeze(), cmap='gray', vmax=1, vmin=0)
        plt.show()
        torch.save(discriminator.state_dict(),
                   proj_path + f"GAN_models\\5x5_Experimental_dis_1.pt")
        torch.save(model.state_dict(),
                   proj_path + f"GAN_models\\5x5_Experimental_gen_1.pt")


def validate_gan(data_path):
    train_transforms_list = [transforms.ToTensor(),
                             transforms.ToPILImage()]
    train_transforms = transforms.Compose(train_transforms_list)
    data_train = ImageDataset(DeblurDataset(data_path).get_paths().data_train, transform=train_transforms)
    data_train_loader = DataLoader(data_train, batch_size=64, shuffle=False, num_workers=8)

    model = UNet(5, 1).cuda()
    #    model.load_state_dict(torch.load("C:\\Users\\LukePC\\PycharmProjects\\snake-rl\\GAN_models\\GAN_1.pt"))
    model.eval()
    # for i, img in enumerate(data_train_loader):
    img_blur, img_sharp = validate_prepare_data(data_path)
    print(img_blur.shape)
    print(img_blur.unsqueeze(0).float().cuda().shape)
    img_deblur, reward = model(img_blur.unsqueeze(0).float().cuda())

    # input_image = img_blur[0][0].detach().cpu().numpy().squeeze()
    predicted_output_img = img_deblur[0].detach().cpu().numpy().squeeze()
    # actual_output = img_sharp[0].detach().cpu().numpy().squeeze()
    # plt.imshow(input_image)
    # plt.show()
    plt.imshow(predicted_output_img)
    plt.show()
    # plt.imshow(actual_output)
    # plt.show()


if __name__ == '__main__':
    bla = 1
    data_path = "D:\\ProjPickleDump\\images_5x5_py3-7"
    data_path2 = "D:\\ProjPickleDump\\images_5x5_py3-7\\train_reward"
    # train,val = DeblurDataset(data_path).get_paths()

    train_gan()
    
    validate_gan(data_path)
    # balance_files()
    #train_reward_model(data_path)
    # experimental_train()
