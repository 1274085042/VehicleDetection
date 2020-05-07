import write_dataset.dataset_to_tfrecords as CT

if __name__=="__main__":
    CT.run("./Images/","Images/tfrecords/","train")