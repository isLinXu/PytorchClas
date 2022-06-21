# PyTorch Image Classifier

this is a standared, clean, easy to read implementation of training a image classifier. It targets on newbie who want get into deeplearning.

I maintained this repo and try my best to answer every question people ask me.

As for a deep learning project, 3 things must keep in mind:

1. Your model must be separate with data processing, which means, **do not** mess your model with any operation with data (except tensor variables);
2. Using a dataloader (tfrecords I am *not* recommend);
3. Writing resuming training logic and track keyboard interuption event.

This example show exactly above philosophy and I hope every people keep that in mind (otherwise your code would be hard to read and messed like a shit).

**keep clean and elegant**.

## Run

To run this code, simply:

```
cd data
# download data and unzip it
./get_flowers.sh
tar -xvf flowers.tar.gz
cd ..
python3 classifier.py
```

You can resume training and keyboard interrupt at any time. that is our philosophy.

