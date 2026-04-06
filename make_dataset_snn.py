import tonic
import torch
import torchvision
from torchdata.datapipes.iter import Mapper
from tonic import DiskCachedDataset

def make_dataset_snn(dataset, use_time_window=False, time_window=10, num_time_steps=10, label_percent=100, 
                     batch_size_train=128, batch_size_test=128, num_workers=8, resize_dim=128):

    if dataset == 'cifar10dvs':

        sensor_size = (128, 128, 2)

        time_window = 1000*time_window

        def data_transform_time_window(data):

            event_tensor = tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=time_window)
            resize = torchvision.transforms.Compose([torch.tensor, torchvision.transforms.Resize((resize_dim, resize_dim))])
            resize_events = resize(event_tensor(data))

            return resize_events 
        
        def data_transform_num_steps(data):

            event_tensor = tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=num_time_steps)
            resize = torchvision.transforms.Compose([torch.tensor, torchvision.transforms.Resize((resize_dim, resize_dim))])
            resize_events = resize(event_tensor(data))

            return resize_events

        if use_time_window:
            data_transform = data_transform_time_window
        else:
            data_transform = data_transform_num_steps

        full_set = tonic.datasets.CIFAR10DVS(save_to='/raid/ee-udayan/uganguly/raghav/ssl_2/data', transform=data_transform)

        # make 90,10 split for train and test
        full_trainset, testset = torch.utils.data.random_split(full_set, [9000, 1000],  generator=torch.Generator().manual_seed(42))

        if label_percent == 100:
            trainset = full_trainset
        elif label_percent != 100:
            trainset, _ = torch.utils.data.random_split(full_trainset, [int(label_percent*len(full_trainset)/100), len(full_trainset) - int(label_percent*len(full_trainset)/100)],
                                                        generator=torch.Generator().manual_seed(42))

        print("trainset: ", len(trainset))
        print("testset: ", len(testset))

        for sample in trainset:
            data, target = sample
            print("sample", sample[0].dtype)
            print("data.shape", data.shape)
            print(target)
            break

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True, num_workers=num_workers, 
                                                drop_last=False, collate_fn=tonic.collation.PadTensors(batch_first=True))
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=True, num_workers=num_workers, 
                                                drop_last=False, collate_fn=tonic.collation.PadTensors(batch_first=True))
        
    elif dataset == 'dvsgestures':

        sensor_size = (128, 128, 2)
        time_window = 10000*time_window
        resize_dim = resize_dim

        def data_transform_time_window(data):

            event_tensor = tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=time_window)
            resize = torchvision.transforms.Compose([torch.tensor, torchvision.transforms.Resize((resize_dim, resize_dim))])
            resize_events = resize(event_tensor(data))

            return resize_events

        def data_transform_num_steps(data):
            
            event_tensor = tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=num_time_steps)
            resize = torchvision.transforms.Compose([torch.tensor, torchvision.transforms.Resize((resize_dim, resize_dim))])
            resize_events = resize(event_tensor(data))

            return resize_events
        
        if use_time_window:
            data_transform = data_transform_time_window
        else:
            data_transform = data_transform_num_steps

        full_trainset = tonic.datasets.DVSGesture(save_to='/raid/ee-udayan/uganguly/raghav/ssl_2/data', train=True, transform=data_transform)
        testset = tonic.datasets.DVSGesture(save_to='/raid/ee-udayan/uganguly/raghav/ssl_2/data', train=False, transform=data_transform)

        if label_percent == 100:
            trainset = full_trainset
        elif label_percent != 100:
            trainset, _ = torch.utils.data.random_split(full_trainset, [int(label_percent*len(full_trainset)/100), len(full_trainset) - int(label_percent*len(full_trainset)/100)],
                                                        generator=torch.Generator().manual_seed(42))

        print("trainset: ", len(trainset))
        print("testset: ", len(testset))

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True, num_workers=num_workers, 
                                                drop_last=False, collate_fn=tonic.collation.PadTensors(batch_first=True))
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=True, num_workers=num_workers,
                                                drop_last=False, collate_fn=tonic.collation.PadTensors(batch_first=True))
        
    elif dataset == 'ncaltech101':
        
        sensor_size = tonic.datasets.NCALTECH101.sensor_size
        time_window = 1000*time_window
        resize_dim = resize_dim

        def data_transform_time_window(data):

            event_tensor = tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=time_window)
            resize = torchvision.transforms.Compose([torch.tensor, torchvision.transforms.Resize((resize_dim, resize_dim))])
            resize_events = resize(event_tensor(data))

            return resize_events

        def data_transform_num_steps(data):
            
            event_tensor = tonic.transforms.ToFrame(sensor_size=sensor_size, n_time_bins=num_time_steps)
            resize = torchvision.transforms.Compose([torch.tensor, torchvision.transforms.Resize((resize_dim, resize_dim))])
            resize_events = resize(event_tensor(data))

            return resize_events
        
        if use_time_window:
            data_transform = data_transform_time_window
        else:
            data_transform = data_transform_num_steps

        full_set = tonic.datasets.NCALTECH101(save_to='/raid/ee-udayan/uganguly/raghav/ssl_2/data', transform=data_transform)

        full_trainset, testset = torch.utils.data.random_split(full_set, [int(0.9*len(full_set)), 
                    len(full_set) - int(0.9*len(full_set))], generator=torch.Generator().manual_seed(4))

        # convert each unique label to a number from 0 to 100
        for i, label in enumerate(set(full_trainset.dataset.targets)):
            full_trainset.dataset.targets = [i if x == label else x for x in full_trainset.dataset.targets]

        for i, label in enumerate(set(testset.dataset.targets)):
            testset.dataset.targets = [i if x == label else x for x in testset.dataset.targets]

        if label_percent == 100:
            trainset = full_trainset
        elif label_percent != 100:
            trainset, _ = torch.utils.data.random_split(full_trainset, [int(label_percent*len(full_trainset)/100), len(full_trainset) - int(label_percent*len(full_trainset)/100)],
                                                        generator=torch.Generator().manual_seed(42))

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True, num_workers=num_workers,
                                                drop_last=False, collate_fn=tonic.collation.PadTensors(batch_first=True))
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=True, num_workers=num_workers,
                                                drop_last=False, collate_fn=tonic.collation.PadTensors(batch_first=True))

    elif dataset == 'ncars':
        
        SAVE_PATH = "/raid/ee-udayan/uganguly/raghav/ssl_2/data"

        sensor_size = (120, 100, 2)
        time_window = 1000*time_window
        resize_dim = resize_dim

        transform_ncars = tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=time_window)
        data_transform = torchvision.transforms.Compose([transform_ncars, torch.tensor, torchvision.transforms.Resize((resize_dim, resize_dim))])

        full_trainset = tonic.prototype.datasets.NCARS(root=SAVE_PATH, train=True)
        testset = tonic.prototype.datasets.NCARS(root=SAVE_PATH, train=False)

        full_trainset = Mapper(full_trainset, data_transform, input_col=0)
        testset = Mapper(testset, data_transform, input_col=0)

        trainset = full_trainset

        print("trainset: ", len(trainset))
        print("testset: ", len(testset))

        shuf_trainset = torch.utils.data.datapipes.iter.combinatorics.ShufflerIterDataPipe(trainset)

        if label_percent == 100:
            shuf_trainset = shuf_trainset

        elif label_percent != 100:

            class SubsetIterableDataset(torch.utils.data.IterableDataset):
                def __init__(self, dataset, num_samples):
                    self.dataset = dataset
                    self.num_samples = num_samples

                def __iter__(self):
                    iterator = iter(self.dataset)
                    for _ in range(self.num_samples):
                        yield next(iterator)

                def __len__(self):
                    return self.num_samples

            num_samples = int(label_percent * 15422 / 100)
            shuf_trainset = SubsetIterableDataset(shuf_trainset, num_samples)
        
        trainloader = torch.utils.data.DataLoader(shuf_trainset, batch_size=batch_size_train, num_workers=num_workers,
                                                drop_last=False, collate_fn=tonic.collation.PadTensors(batch_first=True))
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, num_workers=num_workers,
                                                drop_last=False, collate_fn=tonic.collation.PadTensors(batch_first=True))

    elif dataset == 'asldvs':

        sensor_size = tonic.datasets.ASLDVS.sensor_size

        time_window = 1000*time_window

        def squeeze_transform(events):
            return events.squeeze()

        def data_transform(data):

            event_transform = tonic.transforms.Compose([
                squeeze_transform,
                tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=time_window),
                ])
            
            resize = torchvision.transforms.Compose([torch.tensor, torchvision.transforms.Resize((224, 224))])

            event_transformed_data = event_transform(data)
            resize_events = resize(event_transformed_data)

            return resize_events
        
        fullset = tonic.datasets.ASLDVS(save_to='./data', transform=data_transform)

        # make 90,10 split for train and test
        full_trainset, testset = torch.utils.data.random_split(fullset, [int(0.9*len(fullset)), len(fullset) - int(0.9*len(fullset))],
                                                            generator=torch.Generator().manual_seed(42))

        if label_percent == 100:
            trainset = full_trainset
        elif label_percent != 100:
            trainset, _ = torch.utils.data.random_split(full_trainset, [int(label_percent*len(full_trainset)/100), len(full_trainset) - int(label_percent*len(full_trainset)/100)],
                                                        generator=torch.Generator().manual_seed(42))

        for sample in trainset:
            data, target = sample
            print("sample", sample[0].dtype)
            print("data.shape", data.shape)
            break

        for sample in testset:
            data, target = sample
            print("sample", sample[0].dtype)
            print("data.shape", data.shape)
            break

        print("trainset: ", len(trainset))
        print("testset: ", len(testset))

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True, num_workers=num_workers, 
                                                drop_last=False, collate_fn=tonic.collation.PadTensors(batch_first=True))
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=True, num_workers=num_workers,
                                                drop_last=False, collate_fn=tonic.collation.PadTensors(batch_first=True))
        
    else:
        raise ValueError('Dataset not found')
    
    return trainloader, testloader