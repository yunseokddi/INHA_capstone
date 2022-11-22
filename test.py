import time
import torch


def test_model(model, device, dataloader, dataset):
    since = time.time()

    dataset_sizes = len(dataset)

    model.eval()

    running_corrects = 0

    # f = open('./14_test_file_name.txt', 'a')
    # for i in range(len(dataloader)):
    #     sample_fname, _ = dataloader.dataset.samples[i]
    #     f.write(str(sample_fname).split('/')[-1] + '\n')

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)

        _, preds = torch.max(outputs, 1)
        # print(preds)
        running_corrects += torch.sum(preds == labels.data)

        # --------- file write ---------
    #     f = open('./14_pred.txt', 'a')
    #     for i in preds:
    #         f.write(str(i.item()) + '\n')
    #
    #     f.close()
    #
    #     f = open('./14_label.txt', 'a')
    #     for i in labels:
    #         f.write(str(i.item()) + '\n')
    #
    # f.close()

    total_acc = running_corrects.double() / dataset_sizes

    print("Num of correct : {}".format(running_corrects.double()))

    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Result of testing Acc: {:4f}'.format(total_acc))

    return True