from generate_data import generative_dataset

if __name__ == '__main__':
    dataset = []
    for i in range(3):
        dataset.append(generative_dataset())
    print(len(dataset))