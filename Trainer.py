import os
import math
import logging
import torch
import torch.optim as opt
import torch.utils.data as ut
import torch.utils.tensorboard as tb

from cosine_annealing import LegacyCosineAnnealingLR
from os import path
from datetime import datetime
from Custom_dataset import Labeled_Unlabeled_dataset as lu
from augmentation import *
from sklearn.model_selection import *
from torch_ema.ema import ExponentialMovingAverage
from tqdm import tqdm , trange


LOGGER_NAME = "Trainer"

class Trainer:

    def __init__(self, dataset, loss_function_X, loss_function_U, batch_size=10, mu=7, use_gpu=True, workers=4):
        '''
        :param data_path: path to the data folder
        :param use_gpu: true if the program should use GPU
        :param data_to_train: percent of data to train
        :param data_to_test: percent of data to test
        :param data_to_eval: percent of data to eval
        '''

        # setup logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(LOGGER_NAME)

        self.mu = mu
        self.logger = logger
        self.dataset = dataset
        self.batch_size = batch_size
        self.loss_function_X = loss_function_X
        self.loss_function_U = loss_function_U
        self.workers = workers

        # setup GPU if possible
        self.main_device = self.get_main_device(use_gpu)

        # setup of tensrobord summary writer, is used to print graphs
        self.summary = tb.SummaryWriter()

    def get_main_device(self, use_gpu):
        '''
           Initilize class by loading data and maybe preprocess
           ASSIGN CUDAS IF POSSIBLE
           :return:
        '''

        # Check if gpu is available
        if torch.cuda.is_available() and use_gpu:
            device = "cuda:0"
            self.logger.info("Using GPU")
            self.check_gpu_card()
        else:
            device = "cpu"
            self.logger.info("Using CPU")

        # assign gpu or cpu to the main_device
        main_device = torch.device(device)

        return main_device

    def check_gpu_card(self):
        '''
        The method tries to check which gpu you are using on your computer
        :return:
        '''
        try:
            import pycuda.driver as cudas
            self.logger.info("The device you are using is: {cudas.Device(0).name()}")

        except ImportError as e:
            self.logger.warn(
                "Could not find pycuda and thus not show amazing stats about youre GPU, have you installed CUDA?")
            pass

    def split_dataset(self, dataset, percent_to_set2):
        assert percent_to_set2 < 1.0
        assert percent_to_set2 > 0

        size_of_dataset = len(dataset)
        length_set1 = int((1.0 - percent_to_set2)*size_of_dataset)
        length_set2 = int(percent_to_set2*size_of_dataset)

        # compensate for integer division
        length_set2 = length_set2 if (length_set1 + length_set2) == len(dataset) else length_set2+1

        set1, set2 = ut.random_split(dataset, [length_set1, length_set2])
        return set1, set2

    def split_labels_per_class(self, dataset, num_labels, num_images = None):

        indices = np.arange(len(dataset))
        labels_indices, unlabeled_indices = train_test_split(indices, train_size=num_labels * self.dataset["num_classes"],
                                                       stratify=dataset.targets)
        if num_images != None:
            labels_indices, unlabeled_indices = self.expand_indicies(labels_indices, unlabeled_indices, num_images)

        return ut.Subset(dataset, labels_indices), ut.Subset(dataset, unlabeled_indices)

    def expand_indicies(self, labeled, unlabeled, num_images):
        print(len(labeled))
        print(num_images)
        expand_label = np.random.choice(labeled, num_images - len(labeled))
        expand_unlabel = np.random.choice(unlabeled, (num_images*self.mu) - len(unlabeled))

        labeled = np.append(labeled, expand_label)
        unlabeled = np.append(unlabeled, expand_unlabel)

        return labeled, unlabeled


    def create_custom_dataloader(self, label, unlabeled):
        '''
        Creates a custom dataset of label and unlabeled
        :param label:
        :param unlabeled:
        :return:
        '''
        # This is a custom dataset located in the file Labeled_Unlabeled_dataset.py
        l_u_dataset = lu.L_U_Dataset(label, unlabeled, self.mu)

        return ut.DataLoader(l_u_dataset, batch_size=self.batch_size, shuffle=True,
                                         num_workers=self.workers, pin_memory=True)

    def create_custom_dataloader2(self, label, unlabeled):
        label_dataloader = ut.DataLoader(label, batch_size=self.batch_size, sampler=ut.RandomSampler(label),
                                         num_workers=self.workers, pin_memory=True, drop_last=True)
        unlabeled_dataloader = ut.DataLoader(unlabeled, batch_size=self.batch_size * self.mu, sampler=ut.RandomSampler(unlabeled),
                                             num_workers=self.workers, pin_memory=True, drop_last=True)

        self.logger.info(f"Labeled length: {len(label_dataloader)}, unlabeled length: {len(unlabeled_dataloader)}")
        return label_dataloader, unlabeled_dataloader


    def validate_directory(self, save_path):
        '''
        Makes sure that the directory under save_path exists, otherwise creates it.
        :param save_path:
        :return:
        '''
        if not path.exists(save_path):
            self.logger.warn("The directory Saved_networks was not found, creating the directory")
            try:
                os.mkdir(save_path)
            except OSError as exc:
                raise

    def save_network(self, model):
        '''
        Saves the model into a directory named Saved_networks
        :param model:
        :return:
        '''
        # create a path to saved networks
        save_path = os.path.dirname("Data")
        save_path = os.path.join(save_path, "Saved_networks")

        # make sure that the directory exists
        self.validate_directory(save_path)

        # create a timestamp
        timestamp = datetime.now()
        timestamp = timestamp.strftime("%d-%m-%Y_%H-%M-%S")

        final_path = save_path + "/" + timestamp
        # save network
        torch.save(model.state_dict(), final_path)

        return final_path

    def save_checkpoint(self, current_state, name):
        save_path = os.path.dirname("Data")
        save_path = os.path.join(save_path, "Saved_checkpoints")
        self.validate_directory(save_path)
        final_path = save_path + "/" + name
        self.logger.info(f"Saving checkpoint at {final_path}")
        torch.save(current_state, final_path)


    def close_summary(self):
        self.summary.flush()
        self.summary.close()

    def log_information(self, learn_rate, weight_decay, momentum, epochs, percent_to_validation, num_labels, checkpoint_ratio, resume_path):
        self.logger.info(f"---------------Training model---------------")
        self.logger.info(f"\t Batch size:\t\t{self.batch_size}")
        self.logger.info(f"\t Mu:\t\t\t{self.mu}")
        self.logger.info(f"\t Learn rate:\t\t{learn_rate}")
        self.logger.info(f"\t Weight decay:\t\t{weight_decay}")
        self.logger.info(f"\t Momentum:\t\t{momentum}")
        self.logger.info(f"\t Epochs:\t\t{epochs}")
        self.logger.info(f"\t Validation percent:\t{percent_to_validation}")
        self.logger.info(f"\t Number of labels:\t{num_labels}")
        self.logger.info(f"\t Checkpoint ratio:\t{checkpoint_ratio}")
        self.logger.info(f"\t Resume path:\t{resume_path}")


    def cosine_learning(self, optimizer, function):
        return opt.lr_scheduler.LambdaLR(optimizer, function)

    def get_cosine_schedule_with_warmup(self, optimizer,
                                        num_warmup_steps,
                                        num_training_steps,
                                        num_cycles=7. / 16.,
                                        last_epoch=-1):
        def _lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            no_progress = float(current_step - num_warmup_steps) / \
                          float(max(1, num_training_steps - num_warmup_steps))
            return max(0., math.cos(math.pi * num_cycles * no_progress))

        return opt.lr_scheduler.LambdaLR(optimizer, _lr_lambda, last_epoch)


    def train(self, model, learn_rate, weight_decay, momentum, num_labels=250, epochs=10, percent_to_validation=0.2,lambda_U=1, threshold=0.95, checkpoint_ratio=None, resume_path=None):
        '''

        :param model:
        :param batch_size:
        :param learn_rate:
        :param weight_decay:
        :param momentum:
        :param epochs:
        :param percent_to_validation:
        :param num_labels: Number of labeled data per class
        :return: a path of the saved model
        '''

        momentum = 0.9
        self.log_information(learn_rate, weight_decay, momentum, epochs, percent_to_validation, num_labels, checkpoint_ratio, resume_path)

        # set model to GPU or CPU
        model.to(self.main_device)

        # split dataset to validation and train, then split train to labeled / unlabeled
        #train, val = self.split_dataset(self.dataset["train_set"], percent_to_validation)
        # The formula represents the percent amount of data to unlabeled data
        #labeled, unlabeled = self.split_dataset(train, self.mu / (1 + self.mu))

        trainset = self.dataset["train_set"]

        self.logger.info(f"Dataset length: {len(trainset)}")

        labeled, unlabeled = self.split_labels_per_class(self.dataset["train_set"], num_labels, num_images=65536) #num_image = 2^16
        #val, unlabeled = self.split_dataset(unlabeled, self.mu / (1+self.mu))

        # Create dataloaders for each part of the dataset
        #train_dataloader = self.create_custom_dataloader(labeled, unlabeled)
        label_dataloader, unlabeled_dataloader = self.create_custom_dataloader2(labeled, unlabeled)

        #unlabeled_dataloader = ut.DataLoader(self.dataset["unlabeled"], batch_size=self.batch_size*self.mu, shuffle=True,
        #                               num_workers=self.workers, pin_memory=True)

        #val_dataloader = ut.DataLoader(val, batch_size=self.batch_size, shuffle=True,
        #                               num_workers=self.workers, pin_memory=True)

        val_dataloader = ut.DataLoader(self.dataset["test_set"], batch_size=self.batch_size, shuffle=True,
                                        num_workers=self.workers, pin_memory=True)
        '''
        TO VERIFY NUMBER OF LABELS
        store = np.zeros(10)
        for j, (X, U) in enumerate(train_dataloader):
            batch_X, label_X = X
            for e in range(len(label_X)):
                store[label_X[e]] += 1

        print(store)
        print(len(train_dataloader))
        print(len(val_dataloader))
        exit(-1)
        '''
        # select optimizer type, current is SGD
        optimizer = opt.SGD(model.parameters(), lr=learn_rate, weight_decay=weight_decay, momentum=momentum, nesterov=True)
        self.ema = ExponentialMovingAverage(model.parameters(), decay=0.995)

        #K total number of steps
        #Weight decay = cos(7*pi*k/(16K)) where k is current step and K total nr of steps
        #K = min(len(label_dataloader), len(unlabeled_dataloader))*(self.batch_size + self.batch_size*self.mu) * epochs
        K = 1048576 #(2^20)
        #scheduler = LegacyCosineAnnealingLR(optimizer, 16*epochs/7)

        cosin = lambda k: max(0., math.cos(7. * math.pi * k / (16. * K)))
        #scheduler = self.cosine_learning(optimizer, cosin)
        scheduler= self.get_cosine_schedule_with_warmup(optimizer,5, K)
        #scheduler = WarmupCosineLrScheduler(
        #    optimizer, max_iter=K, warmup_iter=0
        #)
        start_epoch = 0

        # Load checkpoint
        if resume_path != None:
            if os.path.isfile(resume_path):
                self.logger.info(f"RESUMING network")
                #model, optimizer, start_epoch, scheduler = self.load_checkpoint(resume_path, model, optimizer, scheduler)
                checkpoint = torch.load(resume_path)
                model.load_state_dict(checkpoint['network'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
                start_epoch = checkpoint['epoch']
            else:
                self.logger.warn(f"CANT LOAD CHECKPOINT")
                exit()

        # set the wanted loss function to criterion
        criterion_X = self.loss_function_X
        criterion_U = self.loss_function_U

        model.train()
        for e in range(start_epoch, epochs):
            self.logger.info(f"Epoch {e} of {epochs}")

            for session in ["training", "validation"]:
                if session == "training":
                    current_dataloader = zip(label_dataloader, unlabeled_dataloader)
                    length = min(len(label_dataloader), len(unlabeled_dataloader))
                    model.train()
                else:
                    current_dataloader = val_dataloader
                    length = len(val_dataloader)
                    model.eval()
                combined_loss = 0
                combined_loss_x = 0
                combined_loss_u = 0
                i = 0
                pbar = tqdm(total=length)
                for j, (X, U) in enumerate(current_dataloader):
                    if session == "training":
                        batch_X, label_X = X
                        weak_a, strong_a = U

                        #self.imshow(torchvision.utils.make_grid(batch_X))
                        #self.imshow(torchvision.utils.make_grid(weak_a))
                        #print('GroundTruth: ', label_X)
                    else:
                        # Verification have no unlabeled dataset
                        batch_X, label_X = (X, U)


                    # Send sample and label to GPU or CPU
                    batch_X = batch_X.to(device=self.main_device)

                    label_X = label_X.to(device=self.main_device)

                    # Empty cuda cache to avoid memory issues
                    torch.cuda.empty_cache()

                    if session == "training":
                        # Reset gradients between training
                        optimizer.zero_grad()
                        out_X = model(batch_X)
                        loss_X = criterion_X(out_X, label_X)

                        loss_X.detach()
                        label_X.detach()

                        # remove from vram
                        del batch_X

                        #input_U_wa = weak_augment(batch_U).to(device=self.main_device)
                        weak_a = weak_a.to(device=self.main_device)
                        out_U_wa = model(weak_a)

                        del weak_a

                        with torch.no_grad():
                            # calc classification of wa data and detach the calculation from the training
                            pseudo_labels = torch.softmax(out_U_wa, dim=1)

                            # remove from vram
                            del out_U_wa
                            # take out the highest values for each class and create a mask
                            probs, labels_U = torch.max(pseudo_labels, dim=1)

                            mask = probs.ge(threshold).float()

                        #count the number of unlabel images that will affect the loss
                        num_of_pseudo_labels = torch.nonzero(mask,as_tuple=False)
                        i+=len(num_of_pseudo_labels)    
                        
                        #input_U_sa = strong_augment(batch_U).to(device=self.main_device)
                        strong_a = strong_a.to(device=self.main_device)
                        out_U_sa = model(strong_a)
                        loss_U = torch.mean(criterion_U(out_U_sa, labels_U) * mask)

                        # remove from vram
                        del strong_a
                        del out_U_sa

                        loss_U.detach()

                        combined_loss_x += loss_X.item()
                        combined_loss_u += loss_U.item()

                        loss = loss_X + lambda_U * loss_U
                    else:
                        # Disable gradient modifications
                        with torch.no_grad():
                            out = model(batch_X)

                            loss = criterion_X(out, label_X)

                    combined_loss += loss.item()

                    if session == "training":
                        # Backprop
                        loss.backward()
                        optimizer.step()
                        scheduler.step()
                        self.ema.update(model.parameters())
                    if (i % 1000 == 0):
                        self.logger.info(f"{session} img: {i}")

                    i += label_X.size(0)
                    pbar.update(1)

                combined_loss /= i
                combined_loss_x /= i
                combined_loss_u /= i

                self.logger.info(f"{session} loss: {combined_loss} loss_x: {combined_loss_x} loss_u: {combined_loss_u}")
                self.summary.add_scalar('Loss/' + session, combined_loss, e)
                self.summary.add_scalar('Loss_x/' + session, combined_loss_x, e)
                self.summary.add_scalar('Loss_u/' + session, combined_loss_u, e)

            if e % checkpoint_ratio == 0:
                checkpoint = {
                    'epoch': e+1,
                    'network': model.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                self.save_checkpoint(checkpoint, self.dataset["name"]+"_mu="+str(self.mu)+"_batch="+str(self.batch_size)+"_epoch="+str(e)+".pt.tar")

        return self.save_network(model)

    def imshow(self, img):
        import matplotlib.pyplot as plt
        #img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def test(self, save_path, model):
        '''
        A test method that loads the network provided in save_path to the model in "model".
        :param save_path:
        :param model:
        :return:
        '''
        test_dataloader = ut.DataLoader(self.dataset["test_set"], batch_size=self.batch_size, shuffle=True,
                                        num_workers=self.workers, pin_memory=True)

        model.load_state_dict(torch.load(save_path))

        number_of_testdata = 0
        correct = 0
        for data in test_dataloader:
            sample, label = data

            # Send sample and label to GPU or CPU
            sample = sample.to(device=self.main_device, dtype=torch.float32)
            label = label.to(device=self.main_device, dtype=torch.float32)
            
            # Get parameters from EMA
            self.ema.copy_to(model.parameters())
            with torch.no_grad():
                out = model(sample)
                _, pred = torch.max(out, 1)

                for v in (pred == label):
                    if v:
                        correct += v.sum().item()
                number_of_testdata += label.size(0)

        self.logger.info(f"Accuracy: {(correct / number_of_testdata) * 100}")


class WarmupCosineLrScheduler(opt.lr_scheduler._LRScheduler):
    '''
            This is different from official definition, this is implemented according to
            the paper of fix-match
            '''

    def __init__(
            self,
            optimizer,
            max_iter,
            warmup_iter,
            warmup_ratio=5e-4,
            warmup='exp',
            last_epoch=-1,
    ):
        self.max_iter = max_iter
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup
        super(WarmupCosineLrScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        ratio = self.get_lr_ratio()
        lrs = [ratio * lr for lr in self.base_lrs]
        return lrs

    def get_lr_ratio(self):
        if self.last_epoch < self.warmup_iter:
            ratio = self.get_warmup_ratio()
        else:
            real_iter = self.last_epoch - self.warmup_iter
            real_max_iter = self.max_iter - self.warmup_iter
            ratio = np.cos((7 * np.pi * real_iter) / (16 * real_max_iter))
        return ratio

    def get_warmup_ratio(self):
        assert self.warmup in ('linear', 'exp')
        alpha = self.last_epoch / self.warmup_iter
        if self.warmup == 'linear':
            ratio = self.warmup_ratio + (1 - self.warmup_ratio) * alpha
        elif self.warmup == 'exp':
            ratio = self.warmup_ratio ** (1. - alpha)
        return ratio

'''
Code graveyard

    label_dataloader = ut.DataLoader(label, batch_size=self.batch_size, shuffle=True,
                                                num_workers=self.workers, pin_memory=True)
    unlabeled_dataloader = ut.DataLoader(unlabeled, batch_size=self.batch_size*self.mu, shuffle=True,
                                                num_workers=self.workers, pin_memory=True)

    self.logger.info(f"Labeled {len(label_dataloader)}, Unlabeled {len(unlabeled_dataloader)}")
    assert len(label_dataloader) == len(unlabeled_dataloader)
    return zip(label_dataloader, unlabeled_dataloader)
    
    
     
'''
